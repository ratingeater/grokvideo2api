#!/usr/bin/env python3
"""OpenAI-compatible Grok video bridge.

The bridge is intentionally small:
- accept NewAPI/WeChat friendly video requests;
- normalize Grok video aliases and video parameters;
- forward generation to a Grok2API-compatible backend;
- forward extension to the XianYuDaXian/grok2api /video/extend endpoint;
- keep the existing legacy /videos and xAI fallbacks.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5017
DEFAULT_BACKEND_URL = "http://127.0.0.1:5006/v1"
DEFAULT_GROK2API_CONFIG = "/home/admin/chenyme_grok2api/data/config.toml"
DEFAULT_XAI_KEY_FILE = "/home/admin/.xai_api_keys"
CANONICAL_MODEL = "grok-imagine-1.0-video"
LEGACY_MODEL = "grok-imagine-video"
MODEL_ALIASES = {
    "grok": LEGACY_MODEL,
    "grokvideo": LEGACY_MODEL,
    "grok-video": LEGACY_MODEL,
    "grokimagine": LEGACY_MODEL,
    "grok-imagine": LEGACY_MODEL,
    "grokimaginevideo": LEGACY_MODEL,
    "imaginevideo": LEGACY_MODEL,
    "imagine-video": LEGACY_MODEL,
    LEGACY_MODEL: LEGACY_MODEL,
    CANONICAL_MODEL: CANONICAL_MODEL,
}
SUPPORTED_LENGTHS = {6, 10, 15}
LEGACY_SECONDS = {6, 10, 12, 16, 20}
SUPPORTED_RATIOS = {"3:2", "2:3", "16:9", "9:16", "1:1"}
SUPPORTED_RESOLUTIONS = {"480p", "720p"}
VIDEO_URL_RE = re.compile(r"https?://[^\s<>'\"]+\.(?:mp4|mov|webm)(?:\?[^\s<>'\"]*)?", re.I)
POST_ID_RE = re.compile(r"(?i)(?:post[_-]?id|parent[_-]?post[_-]?id|video[_-]?post[_-]?id)[=:]([0-9a-f-]{16,64})")


class RequestError(RuntimeError):
    def __init__(self, message: str, status: int = 502, code: str = "upstream_error", param: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code
        self.param = param


@dataclass
class NormalizedVideoRequest:
    model: str
    upstream_model: str
    prompt: str
    video_length: int
    aspect_ratio: str
    resolution: str
    preset: str
    n: int
    stream: bool
    extend_post_id: str
    original_post_id: str
    video_extension_start_time: float | None
    stitch_with_extend: bool
    raw: dict[str, Any]


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def load_api_key(config_path: str) -> str:
    for name in ("GROKVIDEO_API_KEY", "GROK2API_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    path = Path(config_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'''(?m)^\s*api_key\s*=\s*["']([^"']+)["']''', text)
    return match.group(1).strip() if match else ""


def load_xai_keys(key_file: str) -> list[str]:
    raw = os.getenv("GROKVIDEO_XAI_KEYS", "").strip() or os.getenv("XAI_API_KEYS", "").strip()
    path = Path(key_file)
    if not raw and path.exists():
        raw = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    keys: list[str] = []
    seen: set[str] = set()
    for item in raw.replace(",", "\n").splitlines():
        key = item.strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def normalize_model(value: Any) -> str:
    raw = str(value or LEGACY_MODEL).strip()
    key = raw.lower().replace(" ", "")
    model = MODEL_ALIASES.get(key, raw)
    if model not in {LEGACY_MODEL, CANONICAL_MODEL}:
        raise RequestError(f"unsupported model: {raw}", status=400, code="unsupported_model", param="model")
    return model


def as_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def nearest_int(value: int, allowed: set[int]) -> int:
    if value in allowed:
        return value
    return min(allowed, key=lambda item: abs(item - value))


def as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def pick_value(payload: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    video_config = payload.get("video_config")
    if isinstance(video_config, dict):
        for name in names:
            if name in video_config and video_config[name] is not None:
                return video_config[name]
    return default


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return ""
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"} and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
                elif isinstance(item.get("content"), str):
                    pieces.append(item["content"])
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(piece for piece in pieces if piece)
    return ""


def extract_prompt(payload: dict[str, Any]) -> str:
    for key in ("prompt", "input", "query"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = payload.get("messages")
    if isinstance(messages, list):
        pieces: list[str] = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") in {"user", "system", "developer"}:
                text = text_from_content(msg.get("content"))
                if text.strip():
                    pieces.append(text.strip())
        if pieces:
            return "\n".join(pieces).strip()
    return ""


def extract_post_id(payload: dict[str, Any], prompt: str) -> str:
    value = pick_value(payload, ("extend_post_id", "post_id", "parent_post_id", "video_post_id"), "")
    text = str(value or "").strip()
    if text:
        return text
    match = POST_ID_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def normalize_request(payload: dict[str, Any]) -> NormalizedVideoRequest:
    model = normalize_model(payload.get("model") or LEGACY_MODEL)
    prompt = extract_prompt(payload)
    if not prompt and not extract_post_id(payload, ""):
        raise RequestError("missing prompt", status=400, code="missing_prompt", param="prompt")

    video_length = nearest_int(
        as_int(pick_value(payload, ("video_length", "duration", "seconds"), 6),
               6),
        SUPPORTED_LENGTHS,
    )
    aspect_ratio = str(pick_value(payload, ("aspect_ratio", "aspectRatio"), "3:2")).strip() or "3:2"
    if aspect_ratio not in SUPPORTED_RATIOS:
        raise RequestError(
            f"aspect_ratio must be one of {sorted(SUPPORTED_RATIOS)}",
            status=400,
            code="invalid_aspect_ratio",
            param="aspect_ratio",
        )
    resolution = str(pick_value(payload, ("resolution", "resolution_name", "resolutionName"), "480p")).strip() or "480p"
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise RequestError(
            "resolution must be one of ['480p', '720p']",
            status=400,
            code="invalid_resolution",
            param="resolution",
        )
    preset = str(pick_value(payload, ("preset",), "normal")).strip() or "normal"
    requested_n = as_int(pick_value(payload, ("n", "concurrent"), 1), 1)
    n = max(1, min(4, requested_n))
    stream = as_bool(payload.get("stream"), False)
    extend_post_id = extract_post_id(payload, prompt)
    original_post_id = str(pick_value(payload, ("original_post_id", "originalPostId"), "") or "").strip()
    start_time = as_float_or_none(pick_value(payload, ("video_extension_start_time", "start_time", "startTime"), None))
    stitch = as_bool(pick_value(payload, ("stitch_with_extend", "stitchWithExtend"), True), True)
    return NormalizedVideoRequest(
        model=model,
        upstream_model=CANONICAL_MODEL,
        prompt=prompt,
        video_length=video_length,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        preset=preset,
        n=n,
        stream=stream,
        extend_post_id=extend_post_id,
        original_post_id=original_post_id,
        video_extension_start_time=start_time,
        stitch_with_extend=stitch,
        raw=payload,
    )


def build_chat_payload(req: NormalizedVideoRequest, *, model: str = CANONICAL_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "stream": bool(req.stream),
        "video_config": {
            "aspect_ratio": req.aspect_ratio,
            "video_length": req.video_length,
            "resolution_name": req.resolution,
            "preset": req.preset,
            "n": req.n,
            "concurrent": req.n,
        },
    }


def build_extend_payload(req: NormalizedVideoRequest, *, model: str = CANONICAL_MODEL) -> dict[str, Any]:
    if not req.extend_post_id:
        raise RequestError("missing extend_post_id/post_id", status=400, code="missing_post_id", param="post_id")
    return {
        "model": model,
        "post_id": req.extend_post_id,
        "prompt": req.prompt,
        "video_length": req.video_length,
        "aspect_ratio": req.aspect_ratio,
        "resolution": req.resolution,
        "stream": bool(req.stream),
        "n": req.n,
        "concurrent": req.n,
        "video_extension_start_time": req.video_extension_start_time or 0.0,
        "stitch_with_extend": req.stitch_with_extend,
    }


def size_from_ratio(req: NormalizedVideoRequest) -> str:
    if req.aspect_ratio == "9:16":
        return "720x1280"
    if req.aspect_ratio == "1:1":
        return "1024x1024"
    if req.aspect_ratio == "2:3":
        return "1024x1792"
    if req.aspect_ratio == "3:2":
        return "1792x1024"
    return "1280x720"


def build_legacy_video_fields(req: NormalizedVideoRequest) -> dict[str, str]:
    return {
        "model": LEGACY_MODEL,
        "prompt": req.prompt,
        "seconds": str(nearest_int(req.video_length, LEGACY_SECONDS)),
        "size": size_from_ratio(req),
        "resolution_name": req.resolution,
        "preset": req.preset,
    }


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def multipart_form_body(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"----grokvideo2api-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def request_bytes(
    method: str,
    url: str,
    *,
    api_key: str = "",
    body: bytes | None = None,
    content_type: str = "application/json; charset=utf-8",
    accept: str = "application/json",
    timeout: int = 900,
) -> tuple[int, str, bytes]:
    headers = {"Accept": accept}
    if body is not None:
        headers["Content-Type"] = content_type
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            mime = str(resp.headers.get("content-type") or "")
            return int(resp.status), mime, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        body_text = raw.decode("utf-8", "replace")[:800]
        raise RequestError(f"HTTP {exc.code} from {url}: {body_text or exc.reason}", status=502, code="upstream_http_error") from exc
    except urllib.error.URLError as exc:
        raise RequestError(f"connection failed for {url}: {exc.reason}", status=502, code="upstream_connection_error") from exc


def request_json(
    method: str,
    url: str,
    *,
    api_key: str = "",
    payload: dict[str, Any] | None = None,
    timeout: int = 900,
    accept: str = "application/json",
) -> dict[str, Any]:
    raw_body = json_bytes(payload or {}) if payload is not None else None
    _, _, raw = request_bytes(
        method,
        url,
        api_key=api_key,
        body=raw_body,
        content_type="application/json; charset=utf-8",
        accept=accept,
        timeout=timeout,
    )
    text = raw.decode("utf-8", "replace")
    if accept.startswith("text/event-stream") or text.lstrip().startswith("data:"):
        return {"_sse_text": text}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RequestError(f"invalid JSON from {url}: {text[:500]}", code="invalid_upstream_json") from exc
    if not isinstance(data, dict):
        raise RequestError(f"upstream JSON is not object from {url}", code="invalid_upstream_shape")
    return data


def parse_sse_objects(raw_text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        value = line.strip()
        if not value.startswith("data:"):
            continue
        payload = value[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def chat_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            value = delta.get("content") or delta.get("reasoning_content")
            if isinstance(value, str):
                return value
        message = choice.get("message")
        if isinstance(message, dict):
            value = message.get("content")
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return text_from_content(value)
    return ""


def nested_text_value(obj: Any, keys: set[str]) -> str:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys and isinstance(value, str) and value.strip():
                return value.strip()
        for value in obj.values():
            found = nested_text_value(value, keys)
            if found:
                return found
    if isinstance(obj, list):
        for value in obj:
            found = nested_text_value(value, keys)
            if found:
                return found
    return ""


def nested_number_value(obj: Any, keys: set[str]) -> float | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys and isinstance(value, (int, float)):
                return float(value)
        for value in obj.values():
            found = nested_number_value(value, keys)
            if found is not None:
                return found
    if isinstance(obj, list):
        for value in obj:
            found = nested_number_value(value, keys)
            if found is not None:
                return found
    return None


def extract_video_url(data: dict[str, Any]) -> str:
    if "_sse_text" in data:
        text = "".join(chat_delta_text(chunk) for chunk in parse_sse_objects(str(data["_sse_text"])))
        match = VIDEO_URL_RE.search(text)
        return match.group(0) if match else ""
    url = nested_text_value(data, {"url", "video_url", "download_url", "file_url"})
    if url:
        return url
    text = chat_delta_text(data)
    if not text:
        choices = data.get("choices")
        if isinstance(choices, list):
            text = "".join(chat_delta_text({"choices": [choice]}) for choice in choices if isinstance(choice, dict))
    match = VIDEO_URL_RE.search(text or "")
    return match.group(0) if match else ""


def extract_b64_and_mime(data: dict[str, Any]) -> tuple[str, str]:
    b64_value = nested_text_value(data, {"b64_json", "b64", "base64", "video_b64"})
    mime_type = nested_text_value(data, {"mime_type", "mime", "content_type"}) or "video/mp4"
    return b64_value, mime_type


def infer_post_id(data: dict[str, Any], url: str = "") -> str:
    value = nested_text_value(data, {"post_id", "postId", "video_post_id", "videoPostId", "parent_post_id", "parentPostId"})
    if value:
        return value
    for text in (url, json.dumps(data, ensure_ascii=False)[:3000]):
        match = POST_ID_RE.search(text)
        if match:
            return match.group(1)
    return ""


def download_video(url: str, timeout: int, api_key: str = "") -> tuple[bytes, str]:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "video/*,*/*"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            mime = str(resp.headers.get("content-type") or "video/mp4").split(";")[0].strip() or "video/mp4"
    except Exception as exc:
        raise RequestError(f"download video failed: {type(exc).__name__} {str(exc)[:300]}", code="download_failed") from exc
    if not raw:
        raise RequestError("downloaded video is empty", code="empty_video")
    return raw, mime


def completed_task(
    model: str,
    *,
    source: str,
    url: str = "",
    raw_video: bytes | None = None,
    mime_type: str = "video/mp4",
    post_id: str = "",
    upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = int(time.time())
    task_id = f"{source}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    data_item: dict[str, Any] = {}
    if url:
        data_item["url"] = url
    if raw_video:
        data_item["b64_json"] = base64.b64encode(raw_video).decode("ascii")
        data_item["mime_type"] = mime_type or "video/mp4"
    task = {
        "id": task_id,
        "request_id": task_id,
        "task_id": task_id,
        "object": "video",
        "created": now,
        "created_at": now,
        "completed_at": now,
        "model": model,
        "status": "completed",
        "progress": 100,
        "url": url,
        "video": {"url": url},
        "data": [data_item],
        "post_id": post_id,
        "source": source,
    }
    if upstream is not None:
        task["upstream"] = {"status": "ok", "shape": list(upstream.keys())[:20]}
    return task


def error_payload(message: str, code: str = "error", param: str = "") -> dict[str, Any]:
    err: dict[str, Any] = {"message": message, "code": code}
    if param:
        err["param"] = param
    return {"error": err}


class GrokVideoRuntime:
    def __init__(
        self,
        *,
        backend_url: str,
        api_key: str,
        xai_keys: list[str],
        timeout: int,
        return_b64: bool,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.api_key = api_key
        self.xai_keys = xai_keys
        self.timeout = timeout
        self.return_b64 = return_b64
        self.tasks: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def store(self, task: dict[str, Any]) -> None:
        task_id = str(task.get("id") or task.get("task_id") or "")
        if not task_id:
            return
        with self.lock:
            self.tasks[task_id] = task
            if len(self.tasks) > 200:
                oldest = sorted(self.tasks, key=lambda key: int(self.tasks[key].get("created_at") or 0))[:50]
                for key in oldest:
                    self.tasks.pop(key, None)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.tasks.get(task_id)

    def maybe_embed_url(self, task: dict[str, Any], url: str) -> dict[str, Any]:
        if not self.return_b64 or not url:
            return task
        if task.get("data") and isinstance(task["data"], list):
            first = task["data"][0]
            if isinstance(first, dict) and first.get("b64_json"):
                return task
        try:
            raw, mime_type = download_video(url, min(self.timeout, 300), api_key="")
        except RequestError as exc:
            task["warning"] = str(exc)
            return task
        first = task.setdefault("data", [{}])[0]
        if isinstance(first, dict):
            first["b64_json"] = base64.b64encode(raw).decode("ascii")
            first["mime_type"] = mime_type
        return task

    def convert_upstream_result(self, data: dict[str, Any], req: NormalizedVideoRequest, source: str) -> dict[str, Any]:
        if isinstance(data.get("data"), list):
            b64_value, mime_type = extract_b64_and_mime(data)
            url = extract_video_url(data)
            raw = base64.b64decode(b64_value.split(",", 1)[-1], validate=False) if b64_value else None
            task = completed_task(
                req.model,
                source=source,
                url=url,
                raw_video=raw,
                mime_type=mime_type,
                post_id=infer_post_id(data, url),
                upstream=data,
            )
            return self.maybe_embed_url(task, url)
        b64_value, mime_type = extract_b64_and_mime(data)
        url = extract_video_url(data)
        if not url and not b64_value:
            task_id = nested_text_value(data, {"task_id", "request_id", "id", "video_id"})
            status = nested_text_value(data, {"status", "state"}).lower()
            if task_id and status in {"queued", "created", "pending", "submitted", "in_progress", "processing", "running"}:
                return data
            raise RequestError(f"upstream did not return video url/base64: {json.dumps(data, ensure_ascii=False)[:500]}")
        raw = base64.b64decode(b64_value.split(",", 1)[-1], validate=False) if b64_value else None
        task = completed_task(
            req.model,
            source=source,
            url=url,
            raw_video=raw,
            mime_type=mime_type,
            post_id=infer_post_id(data, url),
            upstream=data,
        )
        return self.maybe_embed_url(task, url)

    def generate_via_chat(self, req: NormalizedVideoRequest, *, model: str) -> dict[str, Any]:
        payload = build_chat_payload(req, model=model)
        data = request_json(
            "POST",
            f"{self.backend_url}/chat/completions",
            api_key=self.api_key,
            payload=payload,
            timeout=self.timeout,
            accept="text/event-stream, application/json" if req.stream else "application/json",
        )
        return self.convert_upstream_result(data, req, f"chat_{model.replace('.', '_').replace('-', '_')}")

    def extend_via_endpoint(self, req: NormalizedVideoRequest, *, model: str) -> dict[str, Any]:
        payload = build_extend_payload(req, model=model)
        data = request_json(
            "POST",
            f"{self.backend_url}/video/extend",
            api_key=self.api_key,
            payload=payload,
            timeout=self.timeout,
            accept="text/event-stream, application/json" if req.stream else "application/json",
        )
        return self.convert_upstream_result(data, req, f"extend_{model.replace('.', '_').replace('-', '_')}")

    def generate_via_legacy_videos_multipart(self, req: NormalizedVideoRequest) -> dict[str, Any]:
        fields = build_legacy_video_fields(req)
        body, content_type = multipart_form_body(fields)
        _, _, raw = request_bytes(
            "POST",
            f"{self.backend_url}/videos",
            api_key=self.api_key,
            body=body,
            content_type=content_type,
            accept="application/json",
            timeout=min(self.timeout, 180),
        )
        created = json.loads(raw.decode("utf-8", "replace"))
        if not isinstance(created, dict):
            raise RequestError("legacy /videos response is not object", code="invalid_legacy_response")
        video_id = str(created.get("id") or created.get("video_id") or "").strip()
        if not video_id:
            raise RequestError(f"legacy /videos missing id: {json.dumps(created, ensure_ascii=False)[:300]}")
        deadline = time.time() + self.timeout
        last_status = str(created.get("status") or "")
        while time.time() < deadline:
            if last_status.lower() in {"completed", "succeeded", "done"}:
                break
            if last_status.lower() in {"failed", "error", "cancelled", "canceled", "expired"}:
                raise RequestError(f"legacy video failed: {json.dumps(created, ensure_ascii=False)[:300]}")
            time.sleep(5)
            created = request_json(
                "GET",
                f"{self.backend_url}/videos/{urllib.parse.quote(video_id, safe='')}",
                api_key=self.api_key,
                timeout=60,
            )
            last_status = str(created.get("status") or "")
        if last_status.lower() not in {"completed", "succeeded", "done"}:
            raise RequestError(f"legacy video timeout id={video_id} status={last_status}", code="video_timeout")
        _, mime_type, raw_video = request_bytes(
            "GET",
            f"{self.backend_url}/videos/{urllib.parse.quote(video_id, safe='')}/content",
            api_key=self.api_key,
            accept="video/*,*/*",
            timeout=180,
        )
        if not raw_video:
            raise RequestError("legacy video content is empty", code="empty_video")
        task = completed_task(
            req.model,
            source="legacy_videos",
            raw_video=raw_video,
            mime_type=(mime_type.split(";")[0].strip() or "video/mp4"),
            post_id=infer_post_id(created),
            upstream=created,
        )
        return task

    def generate_via_xai(self, req: NormalizedVideoRequest, xai_key: str, index: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": LEGACY_MODEL,
            "prompt": req.prompt,
            "duration": req.video_length,
            "resolution": req.resolution,
            "aspect_ratio": req.aspect_ratio,
        }
        data = request_json(
            "POST",
            "https://api.x.ai/v1/videos/generations",
            api_key=xai_key,
            payload=payload,
            timeout=min(self.timeout, 120),
        )
        request_id = str(data.get("request_id") or data.get("id") or data.get("task_id") or "").strip()
        url = extract_video_url(data)
        if not request_id and not url:
            raise RequestError(f"xAI response missing request_id: {json.dumps(data, ensure_ascii=False)[:300]}")
        deadline = time.time() + self.timeout
        last_status = ""
        while request_id and time.time() < deadline:
            time.sleep(10)
            status_data = request_json(
                "GET",
                f"https://api.x.ai/v1/videos/{urllib.parse.quote(request_id, safe='')}",
                api_key=xai_key,
                timeout=60,
            )
            url = extract_video_url(status_data)
            if url:
                data = status_data
                break
            last_status = str(status_data.get("status") or status_data.get("state") or "")
            if last_status.lower() in {"failed", "error", "cancelled", "canceled", "expired"}:
                raise RequestError(f"xAI video failed: {json.dumps(status_data, ensure_ascii=False)[:300]}")
        if not url:
            raise RequestError(f"xAI video timeout request_id={request_id} status={last_status}", code="video_timeout")
        raw_video = None
        mime_type = "video/mp4"
        if self.return_b64:
            raw_video, mime_type = download_video(url, min(self.timeout, 300))
        return completed_task(
            req.model,
            source=f"xai_{index}",
            url=url,
            raw_video=raw_video,
            mime_type=mime_type,
            post_id=infer_post_id(data, url),
            upstream=data,
        )

    def generate(self, req: NormalizedVideoRequest) -> dict[str, Any]:
        if not self.api_key and not self.xai_keys:
            raise RequestError("missing Grok video backend API key", status=500, code="missing_api_key")
        errors: list[str] = []
        if self.api_key:
            for model in (CANONICAL_MODEL, LEGACY_MODEL):
                try:
                    return self.generate_via_chat(req, model=model)
                except RequestError as exc:
                    errors.append(f"chat/{model}: {str(exc)[:500]}")
            try:
                return self.generate_via_legacy_videos_multipart(req)
            except RequestError as exc:
                errors.append(f"legacy /videos: {str(exc)[:500]}")
        for index, xai_key in enumerate(self.xai_keys, start=1):
            try:
                return self.generate_via_xai(req, xai_key, index)
            except RequestError as exc:
                errors.append(f"xAI[{index}]: {str(exc)[:500]}")
        raise RequestError("; ".join(errors)[:1600] or "video generation failed", code="all_backends_failed")

    def extend(self, req: NormalizedVideoRequest) -> dict[str, Any]:
        if not self.api_key:
            raise RequestError("missing Grok video backend API key", status=500, code="missing_api_key")
        errors: list[str] = []
        for model in (CANONICAL_MODEL, LEGACY_MODEL):
            try:
                return self.extend_via_endpoint(req, model=model)
            except RequestError as exc:
                errors.append(f"extend/{model}: {str(exc)[:500]}")
        raise RequestError("; ".join(errors)[:1600] or "video extension failed", code="extend_failed")


class Handler(BaseHTTPRequestHandler):
    server_version = "grokvideo2api/0.1"

    @property
    def runtime(self) -> GrokVideoRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_bytes(self, status: int, raw: bytes, mime_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "models": [LEGACY_MODEL, CANONICAL_MODEL],
                    "backend_url": self.runtime.backend_url,
                    "api_key": bool(self.runtime.api_key),
                    "xai_keys": len(self.runtime.xai_keys),
                },
            )
            return
        if path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": LEGACY_MODEL, "object": "model", "owned_by": "grokvideo2api"},
                        {"id": CANONICAL_MODEL, "object": "model", "owned_by": "grokvideo2api"},
                    ],
                },
            )
            return
        if path.startswith("/v1/videos/") and path.endswith("/content"):
            task_id = path[len("/v1/videos/") : -len("/content")].strip("/")
            task = self.runtime.get(task_id)
            if not task:
                self.send_json(404, error_payload(f"unknown task: {task_id}", "unknown_task"))
                return
            b64_value, mime_type = extract_b64_and_mime(task)
            if not b64_value:
                self.send_json(404, error_payload(f"task has no embedded video content: {task_id}", "missing_content"))
                return
            raw = base64.b64decode(b64_value.split(",", 1)[-1], validate=False)
            self.send_bytes(200, raw, mime_type)
            return
        for prefix in ("/v1/videos/", "/v1/video/generations/"):
            if path.startswith(prefix):
                task_id = path[len(prefix) :].strip("/")
                task = self.runtime.get(task_id)
                if not task:
                    self.send_json(404, error_payload(f"unknown task: {task_id}", "unknown_task"))
                    return
                self.send_json(200, task)
                return
        self.send_json(404, error_payload(f"not found: {self.path}", "not_found"))

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in {"/v1/video/generations", "/v1/videos/generations", "/v1/videos", "/v1/video/extend"}:
            self.send_json(404, error_payload(f"not found: {self.path}", "not_found"))
            return
        try:
            payload = self.read_json()
            req = normalize_request(payload)
            task = self.runtime.extend(req) if path == "/v1/video/extend" or req.extend_post_id else self.runtime.generate(req)
            self.runtime.store(task)
            self.send_json(200, task)
        except RequestError as exc:
            self.send_json(exc.status, error_payload(str(exc), exc.code, exc.param))
        except Exception as exc:
            self.send_json(500, error_payload(f"{type(exc).__name__}: {str(exc)[:500]}", "internal_error"))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RequestError(f"invalid JSON: {exc}", status=400, code="invalid_json") from exc
        if not isinstance(payload, dict):
            raise RequestError("JSON body must be object", status=400, code="invalid_json")
        return payload

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[grokvideo2api] {self.address_string()} {fmt % args}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("GROKVIDEO_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("GROKVIDEO_PORT", str(DEFAULT_PORT))))
    parser.add_argument(
        "--backend-url",
        default=os.getenv("GROKVIDEO_BACKEND_URL", os.getenv("GROK2API_BASE_URL", DEFAULT_BACKEND_URL)),
    )
    parser.add_argument(
        "--config",
        default=os.getenv("GROKVIDEO_CONFIG", os.getenv("GROK2API_CONFIG", DEFAULT_GROK2API_CONFIG)),
    )
    parser.add_argument(
        "--xai-key-file",
        default=os.getenv("GROKVIDEO_XAI_KEY_FILE", os.getenv("XAI_KEY_FILE", DEFAULT_XAI_KEY_FILE)),
    )
    parser.add_argument("--timeout", type=int, default=int(os.getenv("GROKVIDEO_TIMEOUT", "900")))
    parser.add_argument("--no-return-b64", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.config)
    xai_keys = load_xai_keys(args.xai_key_file)
    return_b64 = False if args.no_return_b64 else env_bool("GROKVIDEO_RETURN_B64", True)
    runtime = GrokVideoRuntime(
        backend_url=args.backend_url,
        api_key=api_key,
        xai_keys=xai_keys,
        timeout=args.timeout,
        return_b64=return_b64,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    print(
        "[grokvideo2api] "
        f"listening={args.host}:{args.port} backend={runtime.backend_url} "
        f"api_key={'yes' if api_key else 'no'} xai_keys={len(xai_keys)} return_b64={return_b64}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
