"""Model clients behind one small interface, with an on-disk response cache.

The agent talks in plain JSON (see agent.py), so any model that can read images and follow a JSON format works.
GeminiClient and OpenAICompatClient (for example Qwen2.5-VL served by vLLM) are written against the public SDKs
but have not been run in this repo's CI. Test them on a single protein before a full run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Message:
    role: str  # "user" or "assistant"
    text: str
    images: list[bytes] = field(default_factory=list)


class LLMClient(Protocol):
    name: str

    def generate(self, system: str, messages: list[Message], json_mode: bool = True) -> str: ...


def _key(model: str, system: str, messages: list[Message], json_mode: bool) -> str:
    payload = {
        "model": model,
        "system": system,
        "json": json_mode,
        "messages": [[m.role, m.text, [hashlib.sha256(i).hexdigest() for i in m.images]] for m in messages],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CachedClient:
    """Wrap any client so repeated requests are free and runs are reproducible."""

    def __init__(self, inner: LLMClient, cache_dir: str | Path):
        self.inner = inner
        self.name = inner.name
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.calls = 0

    def generate(self, system, messages, json_mode=True):
        path = self.dir / f"{_key(self.name, system, messages, json_mode)}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["response"]
        self.calls += 1
        out = self.inner.generate(system, messages, json_mode)
        path.write_text(json.dumps({"response": out}))
        return out


def _retry(fn, max_retries=6):
    delay = 4.0
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            transient = any(s in text for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "timed out", "rate"))
            if not transient or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 90)


class GeminiClient:
    def __init__(self, model: str, api_key: str | None = None, temperature: float = 0.0):
        from google import genai  # lazy import

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini client")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.name = f"gemini:{model}"
        self.temperature = temperature

    def generate(self, system, messages, json_mode=True):
        from google.genai import types

        contents = []
        for m in messages:
            parts = [types.Part.from_text(text=m.text)]
            parts += [types.Part.from_bytes(data=b, mime_type="image/png") for b in m.images]
            contents.append(types.Content(role="user" if m.role == "user" else "model", parts=parts))
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = _retry(lambda: self.client.models.generate_content(model=self.model, contents=contents, config=cfg))
        return resp.text or ""


class OpenAICompatClient:
    """Any OpenAI-compatible chat endpoint, such as a local vLLM server hosting Qwen2.5-VL."""

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY", temperature: float = 0.0):
        from openai import OpenAI  # lazy import

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.name = f"openai-compat:{model}"
        self.temperature = temperature

    def generate(self, system, messages, json_mode=True):
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            content = [{"type": "text", "text": m.text}]
            for b in m.images:
                url = "data:image/png;base64," + base64.b64encode(b).decode()
                content.append({"type": "image_url", "image_url": {"url": url}})
            msgs.append({"role": m.role, "content": content})
        resp = _retry(lambda: self.client.chat.completions.create(model=self.model, messages=msgs, temperature=self.temperature))
        return resp.choices[0].message.content or ""


class MockClient:
    """Scripted responses for tests. Records every request."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.name = "mock"
        self.requests: list[tuple[str, list[Message]]] = []

    def generate(self, system, messages, json_mode=True):
        self.requests.append((system, list(messages)))
        return self.responses.pop(0) if self.responses else "{}"
