from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GymRunner:
    def __init__(self, startup_timeout: float = 10.0) -> None:
        self.startup_timeout = startup_timeout
        self._proc: subprocess.Popen | None = None
        self._base_url: str | None = None
        self._module_path: str | None = None

    def start(self, module_path: str) -> str:
        self.shutdown()
        port = _free_port()
        self._module_path = module_path
        project_root = Path(__file__).resolve().parent.parent
        self._proc = subprocess.Popen(
            [sys.executable, "-m", module_path, str(port)],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._base_url = f"http://127.0.0.1:{port}"
        if not self._wait_for_healthz(self._base_url):
            self.shutdown()
            raise RuntimeError(f"app {module_path} did not become healthy within {self.startup_timeout}s")
        return self._base_url

    def _wait_for_healthz(self, base_url: str) -> bool:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False
            try:
                r = httpx.get(f"{base_url}/healthz", timeout=0.5)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        return False

    @property
    def base_url(self) -> str | None:
        return self._base_url

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        except Exception:
            pass
        self._proc = None
        self._base_url = None
        self._module_path = None

    def __del__(self) -> None:
        self.shutdown()
