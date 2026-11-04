import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from noisefloor.client import ChatResponse, Client, Ledger
from noisefloor.proxy import request_from_body, response_body, serve_proxy


class Upstream(BaseHTTPRequestHandler):
    bodies: ClassVar[list[dict[str, Any]]] = []
    status: ClassVar[int] = 200

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.bodies.append(body)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "submit", "arguments": '{"answer": "7"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        }
        raw = json.dumps(payload).encode() if Upstream.status == 200 else b"down"
        self.send_response(Upstream.status)
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def upstream() -> Iterator[str]:
    Upstream.bodies.clear()
    Upstream.status = 200
    httpd = HTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def post(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_request_from_body_translates_fields() -> None:
    req = request_from_body(
        {
            "model": "r1",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "seed": 3,
            "stop": "END",
            "max_completion_tokens": 77,
            "tool_choice": "auto",
            "tools": [{"type": "function"}],
        }
    )
    assert (req.model, req.temperature, req.seed, req.stop, req.max_tokens) == (
        "r1",
        0.5,
        3,
        ("END",),
        77,
    )
    assert req.extra == (("tool_choice", '"auto"'),) and req.tools == ({"type": "function"},)
    plain = request_from_body({"messages": [{"role": "user", "content": "x"}]})
    assert plain.seed is None and plain.max_tokens == 512 and plain.temperature == 0.0
    with pytest.raises(ValueError, match="streaming"):
        request_from_body({"stream": True, "messages": [{"role": "user", "content": "x"}]})
    with pytest.raises(ValueError, match="messages"):
        request_from_body({"messages": []})


def test_response_body_shape() -> None:
    req = request_from_body({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    body = response_body(req, ChatResponse(text="hello", input_tokens=3, output_tokens=2))
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert body["choices"][0]["finish_reason"] == "stop" and body["usage"]["total_tokens"] == 5
    tool = response_body(
        req, ChatResponse(text="", tool_calls=({"id": "1"},), input_tokens=1, output_tokens=1)
    )
    assert tool["choices"][0]["finish_reason"] == "tool_calls"
    assert tool["choices"][0]["message"]["tool_calls"] == [{"id": "1"}]


def test_proxy_routes_through_client(
    tmp_path: Path, upstream: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)
    client = Client(
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        cache_dir=tmp_path / "cache",
        caps={},
        base_urls={"*": upstream},
        label="t",
    )
    body = {
        "model": "run-1",
        "messages": [{"role": "user", "content": "go"}],
        "seed": 5,
        "tool_choice": "auto",
    }
    with serve_proxy(client) as url:
        status, payload = post(f"{url}/v1/chat/completions", body)
        assert status == 200 and payload["model"] == "run-1"
        call = payload["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "submit"
        assert Upstream.bodies[0]["tool_choice"] == "auto" and Upstream.bodies[0]["seed"] == 5
        status2, _ = post(f"{url}/v1/chat/completions", body)
        assert status2 == 200 and len(Upstream.bodies) == 1  # served from the cache
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{url}/v1/models", timeout=5) as resp:
            assert json.loads(resp.read())["data"][0]["id"] == "local"
        with pytest.raises(urllib.error.HTTPError):
            opener.open(f"{url}/nope", timeout=5)
        assert post(f"{url}/v1/other", body)[0] == 404
        assert post(f"{url}/v1/chat/completions", {"messages": []})[0] == 400
        Upstream.status = 500
        assert post(f"{url}/v1/chat/completions", {**body, "seed": 6})[0] == 502
        monkeypatch.setenv("NOISEFLOOR_NO_MODELS", "1")
        assert post(f"{url}/v1/chat/completions", {**body, "seed": 7})[0] == 403
    ledger = [json.loads(ln) for ln in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [e["cached"] for e in ledger] == [False, True]
