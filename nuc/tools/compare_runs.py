from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two NUC prototype summary.json files.")
    parser.add_argument("--left", required=True, help="Path to the first summary.json")
    parser.add_argument("--right", required=True, help="Path to the second summary.json")
    args = parser.parse_args()

    left = load_summary(args.left)
    right = load_summary(args.right)

    print("metric,left,right")
    print(f"frames,{left['runtime']['frames']},{right['runtime']['frames']}")
    print(f"keyframes,{left['runtime']['keyframes']},{right['runtime']['keyframes']}")
    print(f"archives,{left['stats'].get('archives', 0)},{right['stats'].get('archives', 0)}")
    print(f"retrieve_hits,{left['stats'].get('retrieve_hits', 0)},{right['stats'].get('retrieve_hits', 0)}")
    print(f"recoveries,{left['stats'].get('recoveries', 0)},{right['stats'].get('recoveries', 0)}")
    print(
        f"geo_rejects,{left['stats'].get('retrieve_geo_rejects', 0)},{right['stats'].get('retrieve_geo_rejects', 0)}"
    )


if __name__ == "__main__":
    main()
