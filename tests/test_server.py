import unittest

from grokvideo2api import server


class NormalizeVideoRequestTests(unittest.TestCase):
    def test_generation_alias_and_duration(self):
        req = server.normalize_request(
            {
                "model": "grokvideo",
                "prompt": "test video",
                "duration": 10,
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "n": 3,
            }
        )
        self.assertEqual(req.model, "grok-imagine-video")
        self.assertEqual(req.upstream_model, "grok-imagine-1.0-video")
        self.assertEqual(req.video_length, 10)
        self.assertEqual(req.aspect_ratio, "16:9")
        self.assertEqual(req.resolution, "720p")
        self.assertEqual(req.n, 3)

    def test_video_config_aliases(self):
        req = server.normalize_request(
            {
                "model": "grok-imagine-1.0-video",
                "messages": [{"role": "user", "content": "hello"}],
                "video_config": {
                    "video_length": 15,
                    "aspect_ratio": "9:16",
                    "resolution_name": "480p",
                    "concurrent": 2,
                },
            }
        )
        self.assertEqual(req.model, "grok-imagine-1.0-video")
        self.assertEqual(req.prompt, "hello")
        self.assertEqual(req.video_length, 15)
        self.assertEqual(req.aspect_ratio, "9:16")
        self.assertEqual(req.n, 2)

    def test_extend_fields(self):
        req = server.normalize_request(
            {
                "model": "grok-imagine-video",
                "prompt": "continue the shot",
                "extend_post_id": "01234567-89ab-cdef-0123-456789abcdef",
                "start_time": 7.5,
                "duration": 6,
                "stitch_with_extend": False,
            }
        )
        payload = server.build_extend_payload(req)
        self.assertEqual(payload["post_id"], "01234567-89ab-cdef-0123-456789abcdef")
        self.assertEqual(payload["video_extension_start_time"], 7.5)
        self.assertEqual(payload["video_length"], 6)
        self.assertFalse(payload["stitch_with_extend"])

    def test_chat_payload_uses_video_config(self):
        req = server.normalize_request({"model": "grokvideo", "prompt": "city", "duration": 10})
        payload = server.build_chat_payload(req)
        self.assertEqual(payload["model"], "grok-imagine-1.0-video")
        self.assertEqual(payload["video_config"]["video_length"], 10)
        self.assertEqual(payload["video_config"]["n"], 1)


if __name__ == "__main__":
    unittest.main()

