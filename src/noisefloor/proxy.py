"""OpenAI-compatible proxy that sends every request through :class:`noisefloor.client.Client`.

Harnesses that call a model server themselves (Inspect's basic agent) talk to this
proxy instead of llama-server, so their calls get the same ledger lines, cache, and
kill switch as minimal-agent's. It listens on loopback only and supports
``POST /v1/chat/completions`` (non-streaming) and ``GET /v1/models``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from noisefloor.client import (
    BackendError,
    ChatRequest,
    ChatResponse,
    Client,
    ContextLimitError,
    ModelCallRefusedError,
)

PASSTHROUGH = ("tool_choice", "parallel_tool_calls", "response_format", "top_p")


def request_from_body(body: Mapping[str, Any]) -> ChatRequest:
    """Translate an OpenAI chat-completions body into a :class:`ChatRequest`."""
    if body.get("stream"):
        raise ValueError("streaming is not supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    stop = body.get("stop") or ()
    if isinstance(stop, str):
        stop = (stop,)
    extra = tuple(
        (name, json.dumps(body[name], sort_keys=True)) for name in PASSTHROUGH if name in body
    )
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 512
    return ChatRequest(
        backend="openai-local",
        model=str(body.get("model", "")),
        messages=tuple(messages),
        temperature=float(body.get("temperature", 0.0) or 0.0),
        max_tokens=int(max_tokens),
        seed=None if body.get("seed") is None else int(body["seed"]),
        stop=tuple(str(s) for s in stop),
        tools=tuple(body.get("tools") or ()),
        extra=extra,
    )


def response_body(request: ChatRequest, reply: ChatResponse) -> dict[str, Any]:
    """An OpenAI chat-completions response body."""
    text, tool_calls, finish_reason = reply.text, reply.tool_calls, reply.finish_reason
    prompt_tokens, completion_tokens = reply.input_tokens, reply.output_tokens
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    return {
        "id": f"nf-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_handler(client: Client, timeout: float) -> type[BaseHTTPRequestHandler]:
    """A request handler class bound to ``client``."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Mapping[str, Any]) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": "local", "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = request_from_body(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": {"message": f"bad request: {exc}"}})
                return
            try:
                reply = client.chat(request, timeout=timeout)
            except ModelCallRefusedError as exc:
                self._send(403, {"error": {"message": str(exc)}})
                return
            except ContextLimitError as exc:
                # OpenAI's code for this, so the harness sees a context overflow, not a fault.
                error = {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
                self._send(400, {"error": error})
                return
            except BackendError as exc:
                self._send(502, {"error": {"message": str(exc)}})
                return
            self._send(200, response_body(request, reply))

        def log_message(self, *args: object) -> None:
            pass

    return Handler


@contextmanager
def serve_proxy(client: Client, port: int = 0, timeout: float = 600.0) -> Iterator[str]:
    """Run the proxy on loopback for the duration of the block; yields its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(client, timeout))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
