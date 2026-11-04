"""OS sandbox for subject runs (Linux user, network, and mount namespaces).

``sandboxed(cmd, workspace)`` returns a command line that runs ``cmd`` inside new
user, network, and mount namespaces created by ``unshare``. Inside, before the
command starts, :func:`enter` (run as ``python -m noisefloor.sandbox``):

1. brings up the loopback interface (the only network: no route outside);
2. builds a new root on a tmpfs inside the workspace from read-only bind mounts of
   the host's top-level directories, a writable bind of ``workspace``, a fresh
   private ``/tmp`` (with read-only binds of the checkout and its venv when they
   live under the host's /tmp), and ``/dev`` and ``/proc``; then makes that root
   read-only and ``chroot``s into it.

So the command can write only to ``workspace`` and its private ``/tmp``, and can
reach only services it starts itself on loopback. The environment is scrubbed of
tokens, API keys, ``SSH_AUTH_SOCK``, and cloud credentials, with
``GIT_CONFIG_NOSYSTEM=1`` and ``GIT_CONFIG_GLOBAL=/dev/null``.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import shutil
import socket
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 1
SKIP_TOP = {"tmp", "proc", "dev", "sys", "old_root", "lost+found"}
SECRET_PREFIXES = ("AWS_", "AZURE_", "GOOGLE_", "GCLOUD_", "ANTHROPIC_", "OPENAI_", "CLAUDE_")
SECRET_NAMES = {
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "GITHUB_MODELS_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "SSH_AUTH_SOCK",
    "CLAUDECODE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "NO_PROXY",
    "no_proxy",
}


class SandboxUnavailableError(RuntimeError):
    """Raised when this OS cannot provide the namespace sandbox."""


def scrubbed_env(
    base: Mapping[str, str] | None = None, workspace: str | None = None
) -> dict[str, str]:
    """A copy of the environment with secrets and proxies removed."""
    env = {
        k: v
        for k, v in (os.environ if base is None else base).items()
        if k not in SECRET_NAMES
        and not k.startswith(SECRET_PREFIXES)
        and not any(s in k for s in ("TOKEN", "SECRET", "API_KEY", "PASSWORD"))
    }
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["NOISEFLOOR_SANDBOXED"] = "1"
    if workspace is not None:
        env["HOME"] = workspace
        env["TMPDIR"] = "/tmp"
    return env


def available() -> bool:
    """True if unprivileged user, network, and mount namespaces work here."""
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    probe = subprocess.run(
        ["unshare", "--user", "--map-root-user", "--net", "--mount", "true"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def default_ro_paths() -> list[str]:
    """Paths under /tmp that the sandboxed command needs to read (this checkout, its venv)."""
    repo = str(Path(__file__).resolve().parents[2])
    return sorted({p for p in (repo, sys.prefix) if p.startswith("/tmp/")})


def sandboxed(cmd: Sequence[str], workspace: str | Path, ro_paths: Sequence[str] = ()) -> list[str]:
    """Command line that runs ``cmd`` in the sandbox with ``workspace`` writable.

    ``ro_paths`` under /tmp are bound read-only into the sandbox's private /tmp.
    """
    ws = str(Path(workspace).resolve())
    ro_args = [arg for path in ro_paths for arg in ("--ro", str(path))]
    return [
        "unshare",
        "--user",
        "--map-root-user",
        "--net",
        "--mount",
        "--fork",
        sys.executable,
        "-m",
        "noisefloor.sandbox",
        ws,
        *ro_args,
        "--",
        *cmd,
    ]


def run(
    cmd: Sequence[str],
    workspace: str | Path,
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` in the sandbox and capture its output."""
    if not available():
        raise SandboxUnavailableError("user, network, and mount namespaces are not available")
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        sandboxed(cmd, ws, default_ro_paths()),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=scrubbed_env(env, str(ws)),
        check=False,
    )


_libc = ctypes.CDLL(None, use_errno=True)


def _mount(
    src: str | None, dst: str, fstype: str | None, flags: int, data: str | None = None
) -> None:
    rc = _libc.mount(
        src.encode() if src else None,
        dst.encode(),
        fstype.encode() if fstype else None,
        flags,
        data.encode() if data else None,
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mount {dst}: {os.strerror(err)}")


def _loopback_up() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        req = struct.pack("16sH14s", b"lo", 0, b"\0" * 14)
        flags = struct.unpack("16sH14s", fcntl.ioctl(sock, SIOCGIFFLAGS, req))[1]
        fcntl.ioctl(sock, SIOCSIFFLAGS, struct.pack("16sH14s", b"lo", flags | IFF_UP, b"\0" * 14))


def _bind_ro(src: str, dst: str) -> None:
    _mount(src, dst, None, MS_BIND | MS_REC)
    _mount(None, dst, None, MS_BIND | MS_REMOUNT | MS_RDONLY)


def enter(workspace: str, ro_paths: Sequence[str] = ()) -> None:
    """Set up the sandbox inside fresh namespaces (call only as namespace root).

    The new root is assembled on a tmpfs inside the workspace, so the host's /tmp is
    never written; the sandbox gets its own empty /tmp.
    """
    _loopback_up()
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    root = f"{workspace}/.nf-root"
    os.makedirs(root, exist_ok=True)
    _mount("tmpfs", root, "tmpfs", MS_NOSUID | MS_NODEV, "size=16m")
    for name in sorted(os.listdir("/")):
        src = f"/{name}"
        dst = f"{root}/{name}"
        if name in SKIP_TOP:
            if name in ("tmp", "proc", "dev"):
                os.mkdir(dst)
            continue
        if os.path.islink(src):
            os.symlink(os.readlink(src), dst)
        elif os.path.isdir(src):
            os.mkdir(dst)
            _bind_ro(src, dst)
    _mount("/dev", f"{root}/dev", None, MS_BIND | MS_REC)
    _mount("/proc", f"{root}/proc", None, MS_BIND | MS_REC)
    _mount("tmpfs", f"{root}/tmp", "tmpfs", MS_NOSUID | MS_NODEV, "size=1g")
    for path in ro_paths:
        if path.startswith("/tmp/") and os.path.isdir(path):
            os.makedirs(f"{root}{path}", exist_ok=True)
            _bind_ro(path, f"{root}{path}")
    inner_ws = f"{root}{workspace}"
    if not os.path.isdir(inner_ws):
        os.makedirs(inner_ws)
    _mount(workspace, inner_ws, None, MS_BIND)
    _mount(None, root, None, MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV)
    os.chroot(root)
    os.chdir(workspace)


def parse_args(args: Sequence[str]) -> tuple[str, list[str], list[str]] | None:
    """Split ``WS [--ro PATH]... -- CMD...`` into (workspace, ro paths, command)."""
    if "--" not in args:
        return None
    sep = list(args).index("--")
    head, cmd = list(args[:sep]), list(args[sep + 1 :])
    if not head or not cmd:
        return None
    workspace, rest = head[0], head[1:]
    if len(rest) % 2 or any(flag != "--ro" for flag in rest[::2]):
        return None
    return workspace, rest[1::2], cmd


def main(argv: list[str] | None = None) -> int:
    """Entry point inside the namespaces: ``python -m noisefloor.sandbox WS [--ro P] -- cmd``."""
    parsed = parse_args(sys.argv[1:] if argv is None else argv)
    if parsed is None:
        print(
            "usage: python -m noisefloor.sandbox WORKSPACE [--ro PATH]... -- CMD ...",
            file=sys.stderr,
        )
        return 2
    workspace, ro_paths, cmd = parsed
    enter(workspace, ro_paths)
    os.execvp(cmd[0], cmd)
    return 127  # execvp does not return


if __name__ == "__main__":
    sys.exit(main())
