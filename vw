#!/usr/bin/env python3
"""vw: the visual workbench. A language model edits a picture, or makes a new one, by writing the code itself.

    vw photo.jpg "make the car dark green"       edit (a picture is given)
    vw "a heron in reeds at dawn, pen and ink"    draw (no picture)
    vw runs | vw eval ... | vw setup | vw --help

See README.md.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PY = HERE / ".venv" / "bin" / "python"

if sys.argv[1:2] == ["setup"]:
    os.execv("/bin/bash", ["/bin/bash", str(HERE / "setup.sh"), *sys.argv[2:]])

# Always run inside this folder's own environment.
if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
    if not VENV_PY.exists():
        sys.exit("The environment is not set up yet: run `vw setup` once.")
    os.execv(str(VENV_PY), [str(VENV_PY), "-I", str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(HERE / "src"))
from cli import main  # noqa: E402

if __name__ == "__main__":
    main()
