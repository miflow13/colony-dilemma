import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io  # noqa: E402
import json  # noqa: E402
import urllib.request  # noqa: E402

from llm import FakeModel, OllamaModel, parse_choice  # noqa: E402

LETTERS = ["A", "B", "C", "D"]


def test_parse_bare_json():
    assert parse_choice('{"choice": "B", "reason": "Fewer die."}', LETTERS) == ("B", "Fewer die.")


def test_parse_fenced_json():
    text = 'Sure.\n```json\n{"choice": "c", "reason": "x"}\n```'
    assert parse_choice(text, LETTERS) == ("C", "x")


def test_parse_json_in_prose():
    text = 'I think this is right: {"choice": "A", "reason": "honesty"} — done.'
    assert parse_choice(text, LETTERS) == ("A", "honesty")


def test_reject_letter_outside_valid():
    assert parse_choice('{"choice": "E", "reason": "x"}', LETTERS) is None


def test_reject_garbage():
    assert parse_choice("I would choose to be honest.", LETTERS) is None


def test_missing_reason_is_empty_string():
    assert parse_choice('{"choice": "D"}', LETTERS) == ("D", "")


def test_fake_model_answers_scene_prompt_validly():
    model = FakeModel(seed=1)
    msgs = [{"role": "user", "content": "...\nOptions:\nA) x\nB) y\nC) z\nAnswer with JSON."}]
    for _ in range(20):
        parsed = parse_choice(model.chat(msgs), ["A", "B", "C"])
        assert parsed is not None
        assert parsed[0] in "ABC"


def test_fake_model_answers_principles_prompt_with_text():
    model = FakeModel(seed=1)
    out = model.chat([{"role": "user", "content": "What principles will guide you?"}])
    assert "choice" not in out
    assert len(out) > 10


def fake_ollama(monkeypatch, replies):
    """Serve canned JSON per URL path instead of opening a socket."""
    seen = []

    def urlopen(req, timeout=None):
        path = req.full_url.split("11434", 1)[1]
        seen.append((path, json.loads(req.data) if req.data else None))
        return io.BytesIO(json.dumps(replies[path]).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


def test_ollama_provenance_reports_digest_version_and_settings(monkeypatch):
    fake_ollama(monkeypatch, {
        "/api/tags": {"models": [{"name": "other:1b", "digest": "zzz"}, {"name": "qwen3:4b", "digest": "abc123"}]},
        "/api/version": {"version": "0.34.4"},
    })
    model = OllamaModel("qwen3:4b", temperature=0.3, base_url="http://localhost:11434")
    assert model.provenance() == {
        "temperature": 0.3, "model_digest": "abc123", "ollama_version": "0.34.4", "think": None,
    }


def test_ollama_provenance_digest_is_none_for_unlisted_model(monkeypatch):
    fake_ollama(monkeypatch, {"/api/tags": {"models": []}, "/api/version": {"version": "1"}})
    assert OllamaModel("ghost", base_url="http://localhost:11434").provenance()["model_digest"] is None


def test_ollama_chat_keeps_thinking_apart_from_the_answer(monkeypatch):
    seen = fake_ollama(monkeypatch, {
        "/api/chat": {"message": {"role": "assistant", "content": "ok", "thinking": "Let me think."}},
    })
    model = OllamaModel("qwen3:4b", base_url="http://localhost:11434")
    assert model.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert model.last_thinking == "Let me think."
    assert "think" not in seen[0][1]  # we leave the setting to Ollama, and log that we did


def test_ollama_chat_without_thinking_clears_last_thinking(monkeypatch):
    fake_ollama(monkeypatch, {"/api/chat": {"message": {"role": "assistant", "content": "ok"}}})
    model = OllamaModel("m", base_url="http://localhost:11434")
    model.last_thinking = "stale"
    model.chat([{"role": "user", "content": "hi"}])
    assert model.last_thinking is None


def test_fake_model_provenance_is_empty():
    assert FakeModel(seed=1).provenance() == {}
