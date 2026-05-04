#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent / "HMR3D" / "nuc" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hikrobot_mvs_first_ten_frames import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
