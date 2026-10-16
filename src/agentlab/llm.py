from __future__ import annotations

import os
import time
from typing import Protocol

import httpx


class LLMError(RuntimeError):
    pass


Messages = list[dict[str, str]]


class LLM(Protocol):
    def chat(self, system: str, messages: Messages) -> str: ...


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.replace("//0.0.0.0", "//localhost")


class OllamaLLM:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str | None = None,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
        retries: int = 2,
    ):
        self.model = model
        self.retries = retries
        self.host = _normalize_host(host or os.getenv("OLLAMA_HOST") or "http://localhost:11434")
        self._client = client or httpx.Client(timeout=timeout)

    def chat(self, system: str, messages: Messages) -> str:
        for attempt in range(self.retries + 1):
            payload = {
                "model": self.model,
                "stream": False,
                "format": "json",
                # a token cap and a higher temperature on retries break endless repetition
                "options": {"temperature": 0.3 * attempt, "num_predict": 512, "repeat_penalty": 1.1},
                "messages": [{"role": "system", "content": system}, *messages],
            }
            try:
                response = self._client.post(f"{self.host}/api/chat", json=payload)
                response.raise_for_status()
                return response.json()["message"]["content"].strip()
            except httpx.HTTPStatusError as error:
                detail = error.response.text[:300]
                if "repeat limit" in detail and attempt < self.retries:
                    continue
                raise LLMError(f"Ollama returned {error.response.status_code}: {detail}") from error
            except (httpx.HTTPError, KeyError, ValueError) as error:
                raise LLMError(f"request to Ollama at {self.host} failed: {error}") from error
        raise LLMError("Ollama did not return a reply")


class GeminiLLM:
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        retries: int = 4,
        pause: float = 2.0,
    ):
        self.model = model
        self.retries = retries
        self.pause = pause
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise LLMError("set GEMINI_API_KEY to use the gemini model")
        self._client = client or httpx.Client(timeout=timeout)

    @staticmethod
    def _transcript(messages: Messages) -> str:
        names = {"user": "User", "assistant": "Assistant"}
        lines = [f"{names.get(m['role'], m['role'])}: {m['content']}" for m in messages]
        return "\n\n".join(lines) + "\n\nAssistant:"

    def chat(self, system: str, messages: Messages) -> str:
        prompt = self._transcript(messages)
        steps = None
        for attempt in range(self.retries + 1):
            payload = {
                "model": self.model,
                "system_instruction": system,
                "input": prompt,
                "generation_config": {"thinking_level": "low"},
            }
            try:
                response = self._client.post(
                    self.url, json=payload, headers={"x-goog-api-key": self._api_key}
                )
                response.raise_for_status()
                steps = response.json()["steps"]
                break
            except httpx.HTTPStatusError as error:
                body = error.response.text
                status = error.response.status_code
                # Gemini sometimes fails to parse its own JSON output and asks for a retry
                bad_json = status == 400 and "invalid JSON" in body
                # a spent daily quota will not recover within a few seconds
                out_of_quota = status == 429 and "per day" in body.lower()
                busy = status in (429, 500, 502, 503, 504) and not out_of_quota
                if (bad_json or busy) and attempt < self.retries:
                    if bad_json:
                        prompt = f"{self._transcript(messages)}\n\n(Previous attempt failed: {body[:200]})"
                    time.sleep(self.pause * (attempt + 1))
                    continue
                raise LLMError(f"Gemini returned {status}: {body[:200]}") from error
            except (httpx.HTTPError, KeyError, ValueError) as error:
                raise LLMError(f"request to Gemini failed: {error}") from error

        texts = [
            block["text"]
            for step in steps
            if step.get("type") == "model_output"
            for block in step.get("content", [])
            if block.get("type") == "text"
        ]
        if not texts:
            raise LLMError("Gemini response contained no text")
        return "".join(texts).strip()


class ScriptedLLM:
    """Returns prepared replies in order, so tests run without a model."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[tuple[str, Messages]] = []

    def chat(self, system: str, messages: Messages) -> str:
        self.calls.append((system, [dict(m) for m in messages]))
        if not self.replies:
            raise LLMError("scripted replies are used up")
        return self.replies.pop(0)


def get_llm(name: str | None = None, model: str | None = None) -> LLM:
    name = (name or os.getenv("AGENTLAB_LLM") or "ollama").lower()
    model = model or os.getenv("AGENTLAB_MODEL")
    if name == "ollama":
        return OllamaLLM(model or "llama3.2:3b")
    if name == "gemini":
        return GeminiLLM(model or "gemini-3.8-flash")
    raise ValueError(f"unknown llm: {name}")
