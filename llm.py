"""Model backends and answer parsing.

OllamaModel talks to a local Ollama server over HTTP using only the
standard library. FakeModel lets the whole pipeline run with no server, for
tests and dry runs.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request

OPTION_LINE = re.compile(r"^([A-Z])\)\s", re.MULTILINE)


class ModelError(RuntimeError):
    """The model backend failed (network, HTTP error, bad payload)."""


class OllamaModel:
    def __init__(self, model: str, temperature: float = 0.8, base_url: str | None = None, timeout: float = 300):
        self.model = model
        self.temperature = temperature
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        # Reasoning a thinking model returned alongside its last answer.
        self.last_thinking: str | None = None

    @property
    def name(self) -> str:
        return self.model

    def _request(self, path: str, payload: dict | None = None) -> dict:
        """POST `payload` as JSON, or GET when there is none."""
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise ModelError(f"Ollama HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ModelError(f"Could not reach Ollama at {self.base_url}: {e}") from e

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        body = self._request("/api/chat", payload)
        try:
            message = body["message"]
            content = message["content"]
        except (KeyError, TypeError) as e:
            raise ModelError(f"Unexpected Ollama response: {json.dumps(body)[:500]}") from e
        self.last_thinking = message.get("thinking") or None
        return content

    def provenance(self) -> dict:
        """What it takes to tell whether two runs used the same model the
        same way. `think` is the setting sent to Ollama: None means it was
        left to Ollama's default for the model."""
        tags = self._request("/api/tags")
        names = (self.model, f"{self.model}:latest")
        digest = next((m.get("digest") for m in tags.get("models", []) if m.get("name") in names), None)
        return {
            "temperature": self.temperature,
            "model_digest": digest,
            "ollama_version": self._request("/api/version").get("version"),
            "think": None,
        }


class FakeModel:
    """Seeded random chooser. Scene prompts get a valid JSON choice; anything
    else gets a canned text answer."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "fake"

    def provenance(self) -> dict:
        return {}

    def chat(self, messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        letters = OPTION_LINE.findall(prompt)
        if letters:
            return json.dumps({"choice": self.rng.choice(letters), "reason": "Random pick from the fake model."})
        return "I will try to keep as many colonists alive as possible while being honest with them."


_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_choice(text: str, valid_letters: list[str]) -> tuple[str, str] | None:
    """Extract (letter, reason) from a model answer. Returns None if no valid
    choice can be found. Never guesses."""
    if not text:
        return None
    cleaned = text.replace("```json", "").replace("```", "")
    for match in _JSON_OBJECT.finditer(cleaned):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        choice = obj.get("choice")
        if not isinstance(choice, str):
            continue
        letter = choice.strip().upper().rstrip(")").strip()
        if letter in valid_letters:
            reason = obj.get("reason", "")
            return letter, reason if isinstance(reason, str) else str(reason)
    return None
