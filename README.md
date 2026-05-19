# grokvideo2api

Small OpenAI-compatible Grok video bridge for NewAPI and WeChat.

It is intentionally separate from the existing Grok chat/image service. The
service accepts stable video aliases, maps video generation and extension
parameters, and forwards to an existing Grok2API-compatible backend.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/video/generations`
- `POST /v1/videos`
- `POST /v1/video/extend`
- `GET /v1/video/generations/{task_id}`
- `GET /v1/videos/{task_id}`
- `GET /v1/videos/{task_id}/content`

## Model Aliases

- `grok-imagine-video`
- `grokvideo`
- `grok-video`
- `grok-imagine-1.0-video`

All aliases are forwarded to the canonical upstream model
`grok-imagine-1.0-video` where supported, with a legacy fallback to
`grok-imagine-video`.

## Configuration

Environment variables:

- `GROKVIDEO_HOST`, default `127.0.0.1`
- `GROKVIDEO_PORT`, default `5017`
- `GROKVIDEO_BACKEND_URL`, default from `GROK2API_BASE_URL` or `http://127.0.0.1:5006/v1`
- `GROKVIDEO_API_KEY`, default from `GROK2API_API_KEY` or config file
- `GROKVIDEO_CONFIG`, default from `GROK2API_CONFIG`
- `GROKVIDEO_XAI_KEYS`, optional comma/newline-separated xAI API keys
- `GROKVIDEO_XAI_KEY_FILE`, optional file with xAI API keys
- `GROKVIDEO_TIMEOUT`, default `900`
- `GROKVIDEO_RETURN_B64`, default `1`

Supported video parameters:

- `duration`, `video_length`, or `video_config.video_length`: `6`, `10`, `15`
- `aspect_ratio`, `aspectRatio`, or `video_config.aspect_ratio`
- `resolution`, `resolution_name`, or `video_config.resolution_name`
- `n` or `concurrent`: `1` to `4`
- `extend_post_id` or `post_id`
- `video_extension_start_time` or `start_time`
- `stitch_with_extend`

## Example

```bash
GROKVIDEO_BACKEND_URL=http://127.0.0.1:5006/v1 \
GROKVIDEO_API_KEY=sk-... \
python -m grokvideo2api.server --host 0.0.0.0 --port 5017
```

```bash
curl -s http://127.0.0.1:5017/v1/video/generations \
  -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{"model":"grok-imagine-video","prompt":"a cat walking through neon rain","duration":10}'
```

