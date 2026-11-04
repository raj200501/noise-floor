"""A fake OpenAI-compatible llama-server for tests (no model; canned submit replies)."""

import sys
from pathlib import Path

SERVER = r"""
import json, sys, http.server
port = int(sys.argv[sys.argv.index("--port") + 1])
ANSWER = sys.argv[sys.argv.index("--answer") + 1] if "--answer" in sys.argv else "nope"
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status": "ok"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        action = json.dumps({"tool": "submit", "args": {"answer": ANSWER}})
        msg = {"role": "assistant", "content": action}
        if body.get("tools"):
            call = {"id": "c1", "type": "function",
                    "function": {"name": "submit", "arguments": json.dumps({"answer": ANSWER})}}
            msg = {"role": "assistant", "content": "", "tool_calls": [call]}
        out = {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
        raw = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def write_fake_server(directory: Path) -> Path:
    """Write the fake llama-server executable and return its path."""
    path = directory / "llama-server"
    path.write_text(f"#!{sys.executable}\n{SERVER}")
    path.chmod(0o755)
    return path
