"""Unit tests for V-plugin endpoint resolution (versatile main/vision discovery).

Deterministic — all network probes are mocked, so this runs without a live
server. Verifies that A4 finds the vision model whether it is loaded in the
GGUF Chatbox main slot (:8080) or a dedicated vision port (:8082).

Run:  py -3 -m unittest plugins.v_plugin.test_endpoint   (from SOC_Ultralight/)
  or: py -3 -m unittest test_endpoint                    (from plugins/v_plugin/)
"""

import importlib.util
import unittest
import unittest.mock as mock
from pathlib import Path

_VP_PATH = Path(__file__).resolve().parent / "v_plugin.py"
_spec = importlib.util.spec_from_file_location("vp_under_test", _VP_PATH)
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)


class _Stub:
    """Minimal stand-in for a VPlugin so we can call _resolve_vlm_url directly."""
    _resolve_vlm_url = vp.VPlugin._resolve_vlm_url

    def __init__(self, url):
        self.cfg = {"vlm_server_url": url}
        self.app = mock.Mock()
        self.app._log = mock.Mock()


class EndpointVisionCapableShapeTests(unittest.TestCase):
    def _fake_models(self, payload):
        """Patch urlopen to return `payload` as the /v1/models JSON body."""
        import io, json
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
        cm.__exit__.return_value = False
        return mock.patch("urllib.request.urlopen", return_value=cm)

    def test_gguf_chatbox_shape_multimodal_true(self):
        payload = {"models": [{"name": "x", "capabilities": ["completion", "multimodal"]}]}
        with self._fake_models(payload):
            self.assertTrue(vp._endpoint_vision_capable("http://localhost:8080/v1/chat/completions"))

    def test_completion_only_is_false(self):
        payload = {"models": [{"name": "x", "capabilities": ["completion"]}]}
        with self._fake_models(payload):
            self.assertFalse(vp._endpoint_vision_capable("http://localhost:8080/v1/chat/completions"))

    def test_openai_shape_vision_true(self):
        payload = {"data": [{"id": "x", "capabilities": ["vision"]}]}
        with self._fake_models(payload):
            self.assertTrue(vp._endpoint_vision_capable("http://localhost:8082/v1/chat/completions"))

    def test_unreachable_is_false(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertFalse(vp._endpoint_vision_capable("http://localhost:9999/v1/chat/completions"))


class ResolveVlmUrlTests(unittest.TestCase):
    def test_main_serves_vision_uses_8080(self):
        with mock.patch.object(vp, "_probe_port", lambda p: p in (8080, 8082)), \
             mock.patch.object(vp, "_endpoint_vision_capable", lambda u: ":8080" in u):
            s = _Stub("http://localhost:8080/v1/chat/completions")
            self.assertEqual(s._resolve_vlm_url(),
                             "http://localhost:8080/v1/chat/completions")

    def test_vision_on_8082_falls_back(self):
        # Main (:8080) is listening but text-only; vision model is on :8082.
        with mock.patch.object(vp, "_probe_port", lambda p: p in (8080, 8082)), \
             mock.patch.object(vp, "_endpoint_vision_capable", lambda u: ":8082" in u):
            s = _Stub("http://localhost:8080/v1/chat/completions")
            self.assertEqual(s._resolve_vlm_url(),
                             "http://localhost:8082/v1/chat/completions")

    def test_no_vision_live_uses_first_listening(self):
        with mock.patch.object(vp, "_probe_port", lambda p: p == 8080), \
             mock.patch.object(vp, "_endpoint_vision_capable", lambda u: False):
            s = _Stub("http://localhost:8080/v1/chat/completions")
            self.assertEqual(s._resolve_vlm_url(),
                             "http://localhost:8080/v1/chat/completions")

    def test_nothing_listening_returns_configured(self):
        with mock.patch.object(vp, "_probe_port", lambda p: False), \
             mock.patch.object(vp, "_endpoint_vision_capable", lambda u: False):
            s = _Stub("http://localhost:8082/v1/chat/completions")
            self.assertEqual(s._resolve_vlm_url(),
                             "http://localhost:8082/v1/chat/completions")


class InferenceAwarenessTests(unittest.TestCase):
    """The graceful-wait layer: /slots parsing + URL derivation are pure logic."""

    def test_parse_slots_processing(self):
        payload = [{"is_processing": True, "n_prompt_tokens_processed": 900,
                    "n_prompt_tokens": 2500}]
        self.assertEqual(vp.Agent4Window._parse_slots(payload), (True, 900, 2500))

    def test_parse_slots_idle(self):
        payload = [{"is_processing": False, "n_prompt_tokens_processed": 0,
                    "n_prompt_tokens": 0}]
        self.assertEqual(vp.Agent4Window._parse_slots(payload), (False, 0, 0))

    def test_parse_slots_unparseable_is_none(self):
        self.assertIsNone(vp.Agent4Window._parse_slots([]))
        self.assertIsNone(vp.Agent4Window._parse_slots("not a list"))
        self.assertIsNone(vp.Agent4Window._parse_slots(None))

    def test_slots_url_derived_from_chat_endpoint(self):
        import types as _t
        win = _t.SimpleNamespace(
            plugin=_t.SimpleNamespace(
                cfg={"vlm_server_url": "http://localhost:8082/v1/chat/completions"}))
        self.assertEqual(vp.Agent4Window._slots_url(win),
                         "http://localhost:8082/slots")


class CopyButtonTests(unittest.TestCase):
    """_copy_last must never be a silent no-op (the operator clicked Copy on a
    window with no last response and got an empty clipboard, no feedback):
    last response if present, else the full window transcript, else an
    explicit 'nothing to copy' line — and clipboard errors must be visible."""

    def _win(self, last_response, window_text):
        import types as _t
        events = []
        history = mock.Mock()
        history.get.return_value = window_text
        w = _t.SimpleNamespace(
            _last_response=last_response,
            _history=history,
            _append_history=lambda tag, msg: events.append((tag, msg)),
        )
        return w, events

    def test_copies_last_response_when_present(self):
        w, events = self._win("Polo", "transcript")
        with mock.patch.object(vp.pyperclip, "copy") as cp:
            vp.Agent4Window._copy_last(w)
        cp.assert_called_once_with("Polo")
        self.assertIn("last response", events[-1][1])

    def test_falls_back_to_window_text_when_no_response(self):
        w, events = self._win("", "Agent 4 ready.\nhello")
        with mock.patch.object(vp.pyperclip, "copy") as cp:
            vp.Agent4Window._copy_last(w)
        cp.assert_called_once_with("Agent 4 ready.\nhello")
        self.assertIn("window text", events[-1][1])

    def test_empty_window_reports_instead_of_silence(self):
        w, events = self._win("", "  \n")
        with mock.patch.object(vp.pyperclip, "copy") as cp:
            vp.Agent4Window._copy_last(w)
        cp.assert_not_called()
        self.assertIn("nothing to copy", events[-1][1])

    def test_clipboard_failure_is_reported(self):
        w, events = self._win("Polo", "")
        with mock.patch.object(vp.pyperclip, "copy",
                               side_effect=RuntimeError("no clipboard")):
            vp.Agent4Window._copy_last(w)
        self.assertEqual(events[-1][0], "err")
        self.assertIn("copy failed", events[-1][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
