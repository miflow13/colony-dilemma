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
    def __init__(self, model: str, temperature: float = 0.8, base_url: str | None = None, timeout: float = 300,
                 num_ctx: int | None = None):
        self.model = model
        self.temperature = temperature
        # Context window in tokens, prompt and reply (thinking included) together.
        # None leaves it to Ollama, which drops the oldest tokens once it is full.
        self.num_ctx = num_ctx
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        # Reasoning a thinking model returned alongside its last answer.
        self.last_thinking: str | None = None
        self.last_tokens: dict[str, int | None] | None = None

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
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        body = self._request("/api/chat", payload)
        try:
            message = body["message"]
            content = message["content"]
        except (KeyError, TypeError) as e:
            raise ModelError(f"Unexpected Ollama response: {json.dumps(body)[:500]}") from e
        self.last_thinking = message.get("thinking") or None
        self.last_tokens = {"prompt": body.get("prompt_eval_count"), "output": body.get("eval_count")}
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
            "num_ctx": self.num_ctx,
            "model_digest": digest,
            "ollama_version": self._request("/api/version").get("version"),
            "think": None,
        }


_API_KEY = re.compile(r"sk-[\w*.-]+")


class OpenAIModel:
    """OpenAI chat completions. The key comes from `api_key` or the
    OPENAI_API_KEY environment variable and never appears in errors."""

    def __init__(self, model: str, temperature: float = 0.8, api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1", timeout: float = 300):
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ModelError("No OpenAI key: set OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_thinking: str | None = None
        self.last_tokens: dict[str, int | None] | None = None
        # The dated model OpenAI says answered, which can differ from the alias asked for.
        self.served_model: str | None = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def _redact(self, text: str) -> str:
        return _API_KEY.sub("[key]", text.replace(self._api_key, "[key]"))

    def chat(self, messages: list[dict[str, str]]) -> str:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({"model": self.model, "messages": messages,
                             "temperature": self.temperature}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise ModelError(self._redact(f"OpenAI HTTP {e.code}: {detail}")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ModelError(self._redact(f"Could not reach OpenAI: {e}")) from None
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelError(self._redact(f"Unexpected OpenAI response: {json.dumps(body)[:500]}")) from None
        usage = body.get("usage") or {}
        self.last_tokens = {"prompt": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
        self.served_model = body.get("model")
        return content or ""

    def provenance(self) -> dict:
        return {"provider": "openai", "temperature": self.temperature}


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
