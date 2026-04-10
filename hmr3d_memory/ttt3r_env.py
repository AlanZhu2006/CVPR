from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TTT3R_ROOT = REPO_ROOT / "third_party" / "TTT3R"
TTT3R_SRC = TTT3R_ROOT / "src"


def bootstrap_ttt3r_imports(weights: str | None = None) -> None:
    for path in (TTT3R_ROOT, TTT3R_SRC):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    if weights:
        weights_parent = str(Path(weights).resolve().parent)
        if weights_parent not in sys.path:
            sys.path.insert(0, weights_parent)

