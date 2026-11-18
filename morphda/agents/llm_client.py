"""
Generic LLM client for MORPH-DA.

API key is always coupled to its configured endpoint to prevent cross-domain
key disclosure. Custom endpoints require their own dedicated key; the Anthropic
key is never sent to a custom URL.

Usage:
    from morphda.agents.llm_client import build_llm_client
    from morphda.agents.langgraph_agent import MorphDaAgent

    # Official Anthropic API
    import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
    llm   = build_llm_client()
    agent = MorphDaAgent(llm=llm)
    result = agent.run(question, tables)

    # Custom compatible endpoint (separate key required)
    llm = build_llm_client(
        base_url="https://api.example.com/v1/messages",
        api_key="your-api-key",
    )

Environment variables:
    ANTHROPIC_API_KEY   key for the official Anthropic endpoint (default)
    MORPH_DA_API_URL    custom endpoint URL (requires MORPH_DA_API_KEY)
    MORPH_DA_API_KEY    key for the custom endpoint
"""

from __future__ import annotations

import importlib
import os
import re
from typing import Any

ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com/v1/messages"


class LLMClient:
    """
    Anthropic-compatible LLM client with strict key-to-endpoint coupling.

    - Default (no base_url): official Anthropic API, uses ANTHROPIC_API_KEY.
    - Custom base_url: requires MORPH_DA_API_KEY; ANTHROPIC_API_KEY is NOT used.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        custom_url = base_url or os.environ.get("MORPH_DA_API_URL", "")

        if custom_url:
            if not custom_url.startswith("https://"):
                raise ValueError(f"Custom endpoint must use HTTPS: {custom_url!r}")
            self._base_url = custom_url
            self._api_key  = api_key or os.environ.get("MORPH_DA_API_KEY", "")
            if not self._api_key:
                raise EnvironmentError(
                    "MORPH_DA_API_URL is set but MORPH_DA_API_KEY is missing.\n"
                    "Set MORPH_DA_API_KEY for the custom endpoint."
                )
        else:
            self._base_url = ANTHROPIC_DEFAULT_URL
            self._api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not self._api_key:
                raise EnvironmentError(
                    "No API key found. Set ANTHROPIC_API_KEY:\n"
                    "  export ANTHROPIC_API_KEY=sk-ant-..."
                )

        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

    def _headers(self) -> dict:
        return {
            "x-api-key":          self._api_key,
            "anthropic-version":  "2023-06-01",
            "content-type":       "application/json",
        }

    def invoke(self, messages: list[dict], **kwargs: Any) -> "LLMResponse":
        """Send a messages request. Compatible with LangChain ChatModel.invoke()."""
        import requests

        system_parts  = [m["content"] for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") != "system"]

        payload: dict[str, Any] = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "messages":   user_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self.temperature != 0.0:
            payload["temperature"] = self.temperature

        resp = requests.post(
            self._base_url,
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle extended-thinking: find first text block
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=data.get("model", self.model),
        )

    def __call__(self, messages: list[dict], **kwargs: Any) -> "LLMResponse":
        return self.invoke(messages, **kwargs)


class LLMResponse:
    """Response compatible with LangChain AIMessage."""

    def __init__(self, content: str, input_tokens: int = 0,
                 output_tokens: int = 0, model: str = "") -> None:
        self.content = content
        self.model   = model

        class _U:
            pass
        u = _U()
        u.input_tokens  = input_tokens
        u.output_tokens = output_tokens
        self.usage_metadata = u

    def __str__(self) -> str:
        return self.content


def model_slug(model: str) -> str:
    """Filesystem-safe model name used for run directories (claude-haiku-4-5 -> claude_haiku_4_5)."""
    return "".join("_" if c in "/-:." else c for c in model)


def build_llm_client(
    model: str = "claude-haiku-4-5",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    provider: str | None = None,
):
    """
    Build an LLM client for MorphDaAgent.

    Provider (argument, else MORPH_DA_PROVIDER, else "anthropic"):
      - "anthropic": official Anthropic API, key = ANTHROPIC_API_KEY
                     (or a custom endpoint via MORPH_DA_API_URL + MORPH_DA_API_KEY)
      - "ollama":    local model served by Ollama, no key needed
      - any other name <p>: loaded from morphda/agents/<p>_client.py, which must define
                     build_client(model, temperature). Use this to plug in your own backend.

    Example:
        export ANTHROPIC_API_KEY=sk-ant-...
        from morphda.agents.llm_client import build_llm_client
        llm = build_llm_client(model="claude-haiku-4-5")
    """
    provider = (provider or os.environ.get("MORPH_DA_PROVIDER") or "anthropic").lower()
    if provider == "anthropic":
        return LLMClient(model=model, api_key=api_key,
                         base_url=base_url, temperature=temperature)
    module_name = f"morphda.agents.{provider}_client"
    if not re.fullmatch(r"[a-z][a-z0-9_]*", provider):
        raise ValueError(f"Invalid provider name {provider!r}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ValueError(f"Unknown provider {provider!r}: no module {module_name}") from exc
    return module.build_client(model=model, temperature=temperature)
