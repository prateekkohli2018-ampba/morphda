"""
Generic LLM gateway client for MORPH-DA.

API key is always coupled to its configured endpoint to prevent cross-domain
key disclosure. Custom endpoints require their own dedicated key; the Anthropic
key is never sent to a custom URL.

Usage:
    from morphda.agents.llm_gateway import build_llm_client
    from morphda.agents.langgraph_agent import MorphDaAgent

    # Official Anthropic API
    import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
    llm   = build_llm_client()
    agent = MorphDaAgent(llm=llm)
    result = agent.run(question, tables)

    # Custom compatible endpoint (separate key required)
    llm = build_llm_client(
        base_url="https://your-proxy.example.com/v1/messages",
        api_key="your-proxy-key",
    )

Environment variables:
    ANTHROPIC_API_KEY   key for the official Anthropic endpoint (default)
    MORPH_DA_API_URL    custom endpoint URL (requires MORPH_DA_API_KEY)
    MORPH_DA_API_KEY    key for the custom endpoint
"""

from __future__ import annotations

import os
from typing import Any

ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com/v1/messages"


class LLMGatewayClient:
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
        self._ca         = os.environ.get("REQUESTS_CA_BUNDLE")

    def _headers(self) -> dict:
        return {
            "x-api-key":          self._api_key,
            "anthropic-version":  "2023-06-01",
            "content-type":       "application/json",
        }

    def invoke(self, messages: list[dict], **kwargs: Any) -> "GatewayResponse":
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
            verify=self._ca,
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
        return GatewayResponse(
            content=content,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=data.get("model", self.model),
        )

    def __call__(self, messages: list[dict], **kwargs: Any) -> "GatewayResponse":
        return self.invoke(messages, **kwargs)


class GatewayResponse:
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


def build_llm_client(
    model: str = "claude-haiku-4-5",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> LLMGatewayClient:
    """
    Build an LLM client for MorphDaAgent.

    Key-endpoint coupling:
      - Default: official Anthropic (api.anthropic.com), key = ANTHROPIC_API_KEY
      - Custom URL: set MORPH_DA_API_URL + MORPH_DA_API_KEY

    Example:
        export ANTHROPIC_API_KEY=sk-ant-...
        from morphda.agents.llm_gateway import build_llm_client
        llm = build_llm_client(model="claude-haiku-4-5")
    """
    return LLMGatewayClient(model=model, api_key=api_key,
                            base_url=base_url, temperature=temperature)
