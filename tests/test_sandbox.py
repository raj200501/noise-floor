import json
import sys
from pathlib import Path

import pytest

from noisefloor import sandbox

needs_ns = pytest.mark.skipif(not sandbox.available(), reason="namespaces unavailable")

PROBE = """
import json, os, socket, urllib.request
out = {"writable": {}, "network": "open", "loopback": False}
for t in ["/", "/etc", "/usr", "/tmp", os.getcwd(), REPO]:
    out["writable"][t] = os.access(t, os.W_OK)
try:
    urllib.request.urlopen("http://example.com", timeout=3)
except Exception:
    out["network"] = "blocked"
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
socket.create_connection(s.getsockname(), 2); out["loopback"] = True
out["secrets"] = [k for k in os.environ if "TOKEN" in k or k.startswith(("ANTHROPIC", "AWS"))]
out["git"] = (os.environ.get("GIT_CONFIG_NOSYSTEM"), os.environ.get("GIT_CONFIG_GLOBAL"))
open("result.json", "w").write(json.dumps(out))
"""


@needs_ns
def test_sandbox_confines_writes_network_and_env(tmp_path: Path) -> None:
    repo = str(Path(__file__).resolve().parents[1])
    ws = Path("/root/.cache/noisefloor/work") if Path("/root").exists() else tmp_path
    ws = ws / f"pytest-{tmp_path.name}"
    env = {"PATH": "/usr/bin:/bin", "GH_TOKEN": "x" * 20, "AWS_SECRET_ACCESS_KEY": "y" * 20}
    code = PROBE.replace("REPO", repr(repo))
    res = sandbox.run([sys.executable, "-c", code], ws, timeout=60, env=env)
    assert res.returncode == 0, res.stderr
    out = json.loads((ws / "result.json").read_text())
    assert out["writable"] == {
        "/": False,
        "/etc": False,
        "/usr": False,
        "/tmp": True,
        str(ws.resolve()): True,
        repo: False,
    }
    assert out["network"] == "blocked" and out["loopback"] is True
    assert out["secrets"] == [] and out["git"] == ["1", "/dev/null"]


def test_scrubbed_env_and_command_line() -> None:
    env = sandbox.scrubbed_env(
        {
            "PATH": "/bin",
            "GH_TOKEN": "t",
            "MY_API_KEY": "k",
            "HTTPS_PROXY": "p",
            "LANG": "C",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "DB_PASSWORD": "p",
        },
        "/w",
    )
    assert env == {
        "PATH": "/bin",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "NOISEFLOOR_SANDBOXED": "1",
        "HOME": "/w",
        "TMPDIR": "/tmp",
    }
    cmd = sandbox.sandboxed(["echo", "hi"], "/w")
    assert cmd[:6] == ["unshare", "--user", "--map-root-user", "--net", "--mount", "--fork"]
    assert cmd[-4:] == ["/w", "--", "echo", "hi"]


def test_main_usage_and_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sandbox.main(["only-one-arg"]) == 2
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailableError):
        sandbox.run(["true"], "/tmp/x")


def test_available_false_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox.sys, "platform", "darwin")
    assert sandbox.available() is False
