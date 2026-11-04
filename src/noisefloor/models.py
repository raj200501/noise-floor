"""Pinned local models and the llama.cpp server that hosts them.

Every local model is pinned by Hugging Face repository, revision, file name, and
the file's SHA-256 (checked before use). Weights live outside the repository in
``NOISEFLOOR_MODELS`` (default ``~/.cache/noisefloor/models``) and are never
committed. The runtime is llama.cpp's ``llama-server`` built from the llama.cpp
sources vendored in llama-cpp-python 0.3.36 (CPU build), found through
``NOISEFLOOR_LLAMA_SERVER``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS_DIR = Path.home() / ".cache" / "noisefloor" / "models"
DEFAULT_SERVER = (
    Path.home()
    / ".cache/noisefloor/src/llama_cpp_python-0.3.36/vendor/llama.cpp/build/bin/llama-server"
)
DEFAULT_CTX = 32768  # tokens per parallel slot (D-018)
NO_THINKING = ("--chat-template-kwargs", '{"enable_thinking": false}')
RUNTIME = {
    "name": "llama.cpp llama-server",
    "source": "llama-cpp-python 0.3.36 sdist (vendor/llama.cpp)",
    "sdist_sha256": "832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e",
    "llama_cpp_commit": "0c1e570",
    "build": "cmake Release, GGML_NATIVE=OFF, CPU only",
}


@dataclass(frozen=True)
class LocalModel:
    """A pinned GGUF model."""

    key: str
    family: str
    params_b: float
    repo: str
    revision: str
    filename: str
    sha256: str
    license: str
    quant: str = "Q4_K_M"
    server_args: tuple[str, ...] = ()

    def path(self, models_dir: Path | None = None) -> Path:
        """Where the weights file is expected."""
        base = models_dir or Path(os.environ.get("NOISEFLOOR_MODELS", DEFAULT_MODELS_DIR))
        return base / self.filename

    def verify(self, models_dir: Path | None = None) -> bool:
        """True if the weights file exists and matches the pinned SHA-256."""
        path = self.path(models_dir)
        if not path.exists():
            return False
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                digest.update(chunk)
        return digest.hexdigest() == self.sha256

    @property
    def url(self) -> str:
        """Pinned download URL."""
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


MODELS: dict[str, LocalModel] = {
    m.key: m
    for m in (
        LocalModel(
            "qwen2.5-0.5b",
            "qwen",
            0.5,
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "9217f5db79a29953eb74d5343926648285ec7e67",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
            "apache-2.0",
        ),
        LocalModel(
            "qwen2.5-1.5b",
            "qwen",
            1.5,
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "91cad51170dc346986eccefdc2dd33a9da36ead9",
            "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
            "apache-2.0",
        ),
        LocalModel(
            "granite-3.3-2b",
            "granite",
            2.5,
            "ibm-granite/granite-3.3-2b-instruct-GGUF",
            "7cdf86ccd1f1bb3491c9b7017b033f2e51367397",
            "granite-3.3-2b-instruct-Q4_K_M.gguf",
            "ac71e9e32c0bea919b409c5918f69ca74339854b0319c5065e4e9fb6d95c4852",
            "apache-2.0",
        ),
        LocalModel(
            "qwen3-1.7b",
            "qwen3",
            1.7,
            "Qwen/Qwen3-1.7B-GGUF",
            "90862c4b9d2787eaed51d12237eafdfe7c5f6077",
            "Qwen3-1.7B-Q8_0.gguf",
            "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
            "apache-2.0",
            quant="Q8_0",
            server_args=NO_THINKING,
        ),
        LocalModel(
            "qwen3-4b",
            "qwen3",
            4.0,
            "Qwen/Qwen3-4B-GGUF",
            "bc640142c66e1fdd12af0bd68f40445458f3869b",
            "Qwen3-4B-Q4_K_M.gguf",
            "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
            "apache-2.0",
            server_args=NO_THINKING,
        ),
        LocalModel(
            "granite-4.0-micro",
            "granite",
            3.0,
            "ibm-granite/granite-4.0-micro-GGUF",
            "ec48475f0c811d812fbfb61975717a9c36eeb652",
            "granite-4.0-micro-Q4_K_M.gguf",
            "97c417dcc0534b0737c74016fb2af083cb17c3b51eaac621192d23961b7024eb",
            "apache-2.0",
        ),
        LocalModel(
            "smollm3-3b",
            "smollm",
            3.0,
            "ggml-org/SmolLM3-3B-GGUF",
            "4965cb60b150737b68a0408c36aeefb65078f894",
            "SmolLM3-Q4_K_M.gguf",
            "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e",
            "apache-2.0",
            server_args=NO_THINKING,
        ),
        LocalModel(
            "phi-4-mini",
            "phi",
            3.8,
            "unsloth/Phi-4-mini-instruct-GGUF",
            "78eb92a46fc37e6b524df991ed9aca9bc6aa7b80",
            "Phi-4-mini-instruct-Q4_K_M.gguf",
            "88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a",
            "mit",
        ),
        LocalModel(
            "smollm2-1.7b",
            "smollm",
            1.7,
            "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
            "2d4a76a30b4af41ecd395c35725ac11688d4cfe4",
            "smollm2-1.7b-instruct-q4_k_m.gguf",
            "decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33",
            "apache-2.0",
        ),
    )
}


def server_binary() -> Path:
    """Path of the llama-server binary."""
    return Path(os.environ.get("NOISEFLOOR_LLAMA_SERVER", DEFAULT_SERVER))


def server_command(
    model: LocalModel,
    port: int,
    threads: int,
    *,
    ctx: int = DEFAULT_CTX,
    parallel: int = 1,
    models_dir: Path | None = None,
) -> list[str]:
    """The exact llama-server command line (recorded in run provenance)."""
    return [
        str(server_binary()),
        "-m",
        str(model.path(models_dir)),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-t",
        str(threads),
        "-c",
        str(ctx * parallel),
        "-np",
        str(parallel),
        "--jinja",
        "--no-webui",
        "--seed",
        "0",
        *model.server_args,
    ]


def wait_ready(port: int, timeout: float = 120.0) -> bool:
    """Poll the server's health endpoint until it answers or the timeout passes."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


@contextmanager
def serve(
    model: LocalModel,
    port: int,
    threads: int,
    *,
    ctx: int = DEFAULT_CTX,
    parallel: int = 1,
    log_path: Path | None = None,
    models_dir: Path | None = None,
) -> Iterator[list[str]]:
    """Run llama-server for the duration of the block; yields its command line."""
    cmd = server_command(model, port, threads, ctx=ctx, parallel=parallel, models_dir=models_dir)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(server_binary().parent)
    log = (log_path or Path(os.devnull)).open("ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_ready(port):
            raise RuntimeError(f"llama-server did not become ready on port {port}")
        yield cmd
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
