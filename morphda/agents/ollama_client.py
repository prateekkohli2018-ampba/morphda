"""
Local-model client for MORPH-DA (Ollama).

Runs the agent against a model served on your own machine: no API key, no
data leaves the host. Used for the zero-cost sample run in the README.

    ollama pull gemma4:e4b
    MORPH_DA_PROVIDER=ollama python scripts/run_natural_agents.py --model gemma4:e4b --max-tasks 5

Environment variables:
    OLLAMA_HOST                  server address (default http://127.0.0.1:11434)
    MORPH_DA_OLLAMA_NUM_CTX      context window in tokens (default 8192)
    MORPH_DA_OLLAMA_TIMEOUT      per-request timeout in seconds (default 600)
"""

from __future__ import annotations

import os
import re
from typing import Any

from morphda.agents.llm_client import LLMResponse

DEFAULT_HOST = "http://127.0.0.1:11434"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _resolve_host(host: str | None) -> str:
    raw = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


class OllamaClient:
    """Chat client for a local Ollama server. Same call interface as LLMClient."""

    def __init__(
        self,
        model: str,
        host: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._url = _resolve_host(host) + "/api/chat"
        self._num_ctx = num_ctx or int(os.environ.get("MORPH_DA_OLLAMA_NUM_CTX", "8192"))
        self._timeout = timeout or float(os.environ.get("MORPH_DA_OLLAMA_TIMEOUT", "600"))

    def invoke(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        import requests

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Skip the reasoning phase of "thinking" models; the agent wants code only.
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self._num_ctx,
                "seed": 0,
            },
        }
        try:
            resp = requests.post(self._url, json=payload, timeout=self._timeout)
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._url}. Start it with 'ollama serve' "
                f"and make sure the model is pulled: ollama pull {self.model}"
            ) from exc
        if resp.status_code == 404:
            raise LookupError(f"Model {self.model!r} not found. Run: ollama pull {self.model}")
        resp.raise_for_status()
        data = resp.json()

        content = _THINK_BLOCK.sub("", data.get("message", {}).get("content", "")).strip()
        return LLMResponse(
            content=content,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=data.get("model", self.model),
        )

    def __call__(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        return self.invoke(messages, **kwargs)


def build_client(model: str, temperature: float = 0.0) -> OllamaClient:
    return OllamaClient(model=model, temperature=temperature)
