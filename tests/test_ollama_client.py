"""OllamaClient against a throwaway local HTTP server (no real model needed)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from morphda.agents.llm_client import build_llm_client, model_slug
from morphda.agents.ollama_client import OllamaClient


class _FakeOllama(BaseHTTPRequestHandler):
    seen: list[dict] = []
    reply: dict = {}
    status: int = 200

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, "body": body})
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(type(self).reply).encode())

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture()
def server():
    _FakeOllama.seen = []
    _FakeOllama.status = 200
    _FakeOllama.reply = {
        "model": "tiny:1b",
        "message": {"role": "assistant",
                    "content": "<think>plan the answer</think>\ndef analyze(tables):\n    return 1"},
        "prompt_eval_count": 120,
        "eval_count": 15,
    }
    httpd = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_returns_code_without_reasoning_and_reports_token_counts(server):
    client = OllamaClient("tiny:1b", host=server)
    reply = client([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])

    assert reply.content == "def analyze(tables):\n    return 1"
    assert (reply.usage_metadata.input_tokens, reply.usage_metadata.output_tokens) == (120, 15)


def test_request_is_deterministic_and_skips_thinking(server):
    OllamaClient("tiny:1b", host=server, temperature=0.0, num_ctx=2048)(
        [{"role": "user", "content": "q"}]
    )
    sent = _FakeOllama.seen[0]
    assert sent["path"] == "/api/chat"
    assert sent["body"]["stream"] is False and sent["body"]["think"] is False
    assert sent["body"]["options"]["temperature"] == 0.0
    assert sent["body"]["options"]["seed"] == 0
    assert sent["body"]["options"]["num_ctx"] == 2048


def test_missing_model_gives_pull_hint(server):
    _FakeOllama.status = 404
    with pytest.raises(LookupError, match="ollama pull tiny:1b"):
        OllamaClient("tiny:1b", host=server)([{"role": "user", "content": "q"}])


def test_unreachable_server_gives_start_hint():
    with pytest.raises(ConnectionError, match="ollama serve"):
        OllamaClient("tiny:1b", host="http://127.0.0.1:9", timeout=2)([{"role": "user", "content": "q"}])


def test_provider_selected_by_env(monkeypatch):
    monkeypatch.setenv("MORPH_DA_PROVIDER", "ollama")
    assert isinstance(build_llm_client(model="tiny:1b"), OllamaClient)


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_llm_client(provider="nope")


def test_provider_name_cannot_import_arbitrary_modules():
    with pytest.raises(ValueError, match="Invalid provider"):
        build_llm_client(provider="os.path;x")


def test_model_slug_keeps_existing_run_directory_names():
    assert model_slug("claude-haiku-4-5") == "claude_haiku_4_5"
    assert model_slug("gemma4:e4b") == "gemma4_e4b"
