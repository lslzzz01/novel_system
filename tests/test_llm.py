from __future__ import annotations

import http.client
import json
import unittest
import urllib.error
from unittest.mock import patch

from novel_agents.llm import OpenAICompatibleClient


class _StreamingResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> "_StreamingResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class _JsonResponse:
    def __init__(self, data: dict) -> None:
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.data


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_invalid_json_is_regenerated_at_low_temperature(self) -> None:
        requests = []

        def urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            content = "not json" if len(requests) == 1 else '{"ok": true}'
            return _JsonResponse(
                {"choices": [{"message": {"content": content}}]}
            )

        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test",
            temperature=1.8,
        )
        with patch("novel_agents.llm.urllib.request.urlopen", side_effect=urlopen):
            result = client.generate_json("返回 JSON", {"task": "test"})

        self.assertEqual({"ok": True}, result)
        self.assertEqual(1.8, requests[0]["temperature"])
        self.assertEqual(0.2, requests[1]["temperature"])
        self.assertIn("output_repair", requests[1]["messages"][1]["content"])

    def test_json_object_can_be_recovered_from_extra_text(self) -> None:
        result = OpenAICompatibleClient._parse_json_object(
            '说明文字\n{"ok": true}\n结束文字'
        )
        self.assertEqual({"ok": True}, result)

    def test_chat_completions_url_appends_v1_when_missing(self) -> None:
        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test",
        )
        self.assertEqual(
            "https://example.test/v1/chat/completions",
            client._chat_completions_url(),
        )
        client.base_url = "https://example.test/v1"
        self.assertEqual(
            "https://example.test/v1/chat/completions",
            client._chat_completions_url(),
        )

    def test_zero_sampling_params_are_omitted(self) -> None:
        captured = {}

        def urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _JsonResponse(
                {"choices": [{"message": {"content": '{"ok": true}'}}]}
            )

        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test/v1",
            temperature=0.8,
            top_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=0,
            seed=0,
        )
        with patch("novel_agents.llm.urllib.request.urlopen", side_effect=urlopen):
            client.generate_json("返回 JSON", {"task": "test"})

        body = captured["body"]
        self.assertEqual(0.8, body["temperature"])
        self.assertNotIn("top_p", body)
        self.assertNotIn("frequency_penalty", body)
        self.assertNotIn("presence_penalty", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("seed", body)

    def test_remote_disconnect_retries_json_then_streaming(self) -> None:
        requests = []

        def urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            requests.append(body)
            # First two non-stream attempts fail with a transient disconnect.
            if not body.get("stream"):
                raise http.client.RemoteDisconnected(
                    "Remote end closed connection without response"
                )
            return _StreamingResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"true}"}}]}\n',
                    b"data: [DONE]\n",
                ]
            )

        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test",
        )
        with (
            patch("novel_agents.llm.urllib.request.urlopen", side_effect=urlopen),
            patch("novel_agents.llm.time.sleep", return_value=None),
        ):
            result = client.generate_json("返回 JSON", {"task": "test"})

        self.assertEqual({"ok": True}, result)
        self.assertGreaterEqual(len(requests), 3)
        self.assertNotIn("stream", requests[0])
        self.assertNotIn("stream", requests[1])
        self.assertTrue(requests[2]["stream"])

    def test_empty_non_stream_response_falls_back_to_streaming(self) -> None:
        requests = []

        def urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            requests.append(body)
            if not body.get("stream"):
                return _JsonResponse({"choices": [{"message": {"content": ""}}]})
            return _StreamingResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"chapter text"}}]}\n',
                    b"data: [DONE]\n",
                ]
            )

        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test",
        )
        with (
            patch("novel_agents.llm.urllib.request.urlopen", side_effect=urlopen),
            patch("novel_agents.llm.time.sleep", return_value=None),
        ):
            result = client.generate_text("write prose", {"task": "chapter_draft"})

        self.assertEqual("chapter text", result)
        self.assertEqual(3, len(requests))
        self.assertTrue(requests[2]["stream"])

    def test_streaming_error_returns_to_non_stream_transport(self) -> None:
        requests = []

        def urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            requests.append(body)
            if len(requests) < 3:
                raise http.client.RemoteDisconnected("connection closed")
            if body.get("stream"):
                return _StreamingResponse(
                    [b'data: {"error":{"type":"upstream_error"}}\n']
                )
            return _JsonResponse(
                {"choices": [{"message": {"content": "recovered text"}}]}
            )

        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test",
        )
        with (
            patch("novel_agents.llm.urllib.request.urlopen", side_effect=urlopen),
            patch("novel_agents.llm.time.sleep", return_value=None),
        ):
            result = client.generate_text("write prose", {"task": "chapter_draft"})

        self.assertEqual("recovered text", result)
        self.assertEqual(
            [False, False, True, False],
            [bool(item.get("stream")) for item in requests],
        )

    def test_local_socket_permission_error_is_not_retried(self) -> None:
        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test",
        )
        permission_error = OSError(10013, "access forbidden")
        with (
            patch(
                "novel_agents.llm.urllib.request.urlopen",
                side_effect=urllib.error.URLError(permission_error),
            ) as urlopen,
            patch("novel_agents.llm.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "WinError 10013"):
                client.generate_text("write prose", {"task": "chapter_draft"})

        self.assertEqual(1, urlopen.call_count)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
