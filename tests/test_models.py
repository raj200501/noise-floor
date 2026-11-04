import hashlib
import socket
import sys
from pathlib import Path

import pytest

from noisefloor import models
from noisefloor.models import MODELS, LocalModel, server_command, wait_ready


def test_registry_is_pinned() -> None:
    assert len({m.family for m in MODELS.values()}) >= 2
    for m in MODELS.values():
        assert len(m.revision) == 40 and len(m.sha256) == 64
        assert m.url.startswith(f"https://huggingface.co/{m.repo}/resolve/{m.revision}/")
        assert m.license in ("apache-2.0", "mit")


def test_verify_checks_hash(tmp_path: Path) -> None:
    (tmp_path / "w.gguf").write_bytes(b"weights")
    good = LocalModel(
        "k", "f", 1.0, "r", "0" * 40, "w.gguf", hashlib.sha256(b"weights").hexdigest(), "mit"
    )
    bad = LocalModel("k", "f", 1.0, "r", "0" * 40, "w.gguf", "0" * 64, "mit")
    missing = LocalModel("k", "f", 1.0, "r", "0" * 40, "nope.gguf", "0" * 64, "mit")
    assert good.verify(tmp_path) and not bad.verify(tmp_path) and not missing.verify(tmp_path)


def test_server_command_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NOISEFLOOR_LLAMA_SERVER", "/opt/llama-server")
    cmd = server_command(MODELS["qwen2.5-0.5b"], 8800, 2, ctx=4096, parallel=2, models_dir=tmp_path)
    assert cmd[0] == "/opt/llama-server" and "--jinja" in cmd
    assert cmd[cmd.index("-c") + 1] == "8192" and cmd[cmd.index("-np") + 1] == "2"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    monkeypatch.setenv("NOISEFLOOR_MODELS", str(tmp_path))
    assert MODELS["qwen2.5-0.5b"].path() == tmp_path / "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def test_serve_with_a_fake_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "llama-server"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import sys, http.server\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
        "    def log_message(self, *a): pass\n"
        "http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("NOISEFLOOR_LLAMA_SERVER", str(fake))
    port = free_port()
    with models.serve(MODELS["qwen2.5-0.5b"], port, 1, log_path=tmp_path / "log") as cmd:
        assert cmd[0] == str(fake) and wait_ready(port, timeout=5)
    assert not wait_ready(port, timeout=1)


def test_serve_raises_when_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("NOISEFLOOR_LLAMA_SERVER", str(fake))
    monkeypatch.setattr(models, "wait_ready", lambda port, timeout=120.0: False)
    with (
        pytest.raises(RuntimeError, match="did not become ready"),
        models.serve(MODELS["qwen2.5-0.5b"], free_port(), 1),
    ):
        pass


def test_server_build_digest_covers_the_shared_libraries(tmp_path) -> None:
    from noisefloor.models import server_build_sha256

    binary = tmp_path / "llama-server"
    assert server_build_sha256(binary) is None
    binary.write_bytes(b"launcher")
    (tmp_path / "libllama.so.0.5.0").write_bytes(b"code v1")
    (tmp_path / "libllama.so.0").symlink_to("libllama.so.0.5.0")
    first = server_build_sha256(binary)
    assert first is not None and len(first) == 64
    (tmp_path / "libllama.so.0.5.0").write_bytes(b"code v2")  # a rebuilt library
    assert server_build_sha256(binary) != first
