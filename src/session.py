"""Starts the sandboxed Python session (worker.py) for one run and talks to it."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

SRC = Path(__file__).resolve().parent
APP = SRC.parent
PROFILE = SRC / "sandbox.sb"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


class SessionError(RuntimeError):
    pass


class Session:
    """One run's Python session. `run(code)` executes code in it and returns what it printed and showed."""

    def __init__(self, work: Path, *, coords: str, sam: bool, timeout: float = 300.0, sandbox: bool = True, ref_level: str = "sample"):
        self.work = Path(work).resolve()
        self.timeout = timeout
        self.sandbox = sandbox
        self.restarts = 0
        self.size: tuple[int, int] | None = None
        (self.work / "tmp").mkdir(parents=True, exist_ok=True)
        # The session's environment is built from nothing: no API key or other secret of the caller reaches it.
        self.env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "HOME": str(self.work),  # whatever a library writes "at home" lands in the run's folder
            "TMPDIR": str(self.work / "tmp"),
            "LANG": "en_US.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(self.work / "tmp"),
            "XDG_CACHE_HOME": str(self.work / "tmp"),
            "NUMBA_CACHE_DIR": str(self.work / "tmp" / "numba"),
            "TORCH_HOME": str(self.work / "tmp"),
            "HF_HUB_OFFLINE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "KIT_COORDS": coords,
            "KIT_SAM": "1" if sam else "0",
            "KIT_REF_LEVEL": ref_level,
            "KIT_MODELS": str(APP / ".models"),
        }
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.start()

    def _command(self) -> list[str]:
        py = [sys.executable, "-I", str(SRC / "worker.py")]
        if not self.sandbox:
            return py
        if not os.path.exists(SANDBOX_EXEC):
            raise SessionError("sandbox-exec is not available on this system, so the model's code cannot be locked into the run's folder. (Only macOS is supported.)")
        params = {
            "HOME": str(Path.home().resolve()),
            "WORK_DIR": str(self.work),
            "VENV_DIR": str(Path(sys.prefix).resolve()),
            "SRC_DIR": str(SRC),
            "MODELS_DIR": str(APP / ".models"),
            "PY_BASE": str(Path(sys.base_prefix).resolve()),
        }
        cmd = [SANDBOX_EXEC, "-f", str(PROFILE)]
        for k, v in params.items():
            cmd += ["-D", f"{k}={v}"]
        return cmd + py

    def start(self) -> None:
        self.lines = queue.Queue()
        self.proc = subprocess.Popen(self._command(), cwd=self.work, env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        proc, lines = self.proc, self.lines

        def pump():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        hello = self._read(120)
        if not hello or not hello.get("ready"):
            err = (hello or {}).get("error") or (proc.stderr.read() if proc.poll() is not None else "no answer")
            raise SessionError(f"the Python session did not start: {str(err)[-1500:]}")
        if hello.get("size") and hello["size"][0]:
            self.size = (hello["size"][0], hello["size"][1])

    def _read(self, timeout: float):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            return {"died": True}
        return json.loads(line)

    def run(self, code: str, *, private: bool = False, timeout: float | None = None) -> dict:
        """Run code. Never raises for what the code does: errors, a timeout or a crash come back in the reply."""
        assert self.proc and self.proc.stdin
        limit = timeout or self.timeout
        try:
            self.proc.stdin.write(json.dumps({"code": code, "private": private}) + "\n")
            self.proc.stdin.flush()
            reply = self._read(limit)
        except (BrokenPipeError, OSError):
            reply = {"died": True}
        if reply is None or reply.get("died"):
            why = f"The code was stopped after {limit:.0f} seconds." if reply is None else "The Python session crashed while running this code (out of memory, or a fault inside a library)."
            self.close()
            self.restarts += 1
            self.start()
            return {"stdout": "", "error": f"{why} The session has been restarted: variables are gone (re-create what you need; `orig`, W, H and the kit are back), files you saved are still there.", "views": [], "dropped_views": 0, "saved": [], "seconds": limit, "restarted": True}
        return reply

    def inventory(self) -> str:
        """The model's variables, one per line."""
        try:
            self.proc.stdin.write(json.dumps({"op": "inventory"}) + "\n")
            self.proc.stdin.flush()
            return (self._read(30) or {}).get("inventory", "(unknown)")
        except Exception:
            return "(unknown)"

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(5)
            except Exception:
                pass
        self.proc = None
