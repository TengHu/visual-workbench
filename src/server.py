"""Keeps one hands-on job's Python session alive between `vw py` commands.

Started by hands.py when a command finds no server for the job; it holds the sandboxed session (session.py) and
answers on a Unix socket in the job's work folder, one request at a time. It ends when told to, or after a long
idle time; the next command then starts a new one (the session's variables are gone, its files are not).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from session import Session  # noqa: E402

SOCK = ".vw.sock"
IDLE = 45 * 60


def main() -> None:
    work = Path(sys.argv[1]).resolve()
    cfg = json.loads((work / "hands.json").read_text())
    os.chdir(work)  # the socket is bound by its short relative name: socket paths are limited to about 100 bytes
    boot = f"{os.getpid()}-{int(time.time())}"
    try:
        session = Session(work, coords=cfg["coords"], sam=cfg["sam"], timeout=cfg["timeout"], sandbox=cfg["sandbox"], ref_level=cfg["ref_level"])
    except Exception as e:  # tell whoever is waiting why there is no session
        (work / ".vw.error").write_text(str(e))
        raise
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(4)
    srv.settimeout(IDLE)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                break
            with conn:
                conn.settimeout(None)
                f = conn.makefile("rw", encoding="utf-8")
                line = f.readline()
                if not line:
                    continue
                req = json.loads(line)
                if req.get("op") == "stop":
                    f.write(json.dumps({"ok": True}) + "\n")
                    f.flush()
                    break
                if req.get("op") == "ping":
                    reply = {"ok": True}
                else:
                    reply = session.run(req["code"], private=bool(req.get("private")), timeout=req.get("timeout"))
                reply["boot"] = boot
                f.write(json.dumps(reply) + "\n")
                f.flush()
    finally:
        session.close()
        srv.close()
        if os.path.exists(SOCK):
            os.unlink(SOCK)


if __name__ == "__main__":
    main()
