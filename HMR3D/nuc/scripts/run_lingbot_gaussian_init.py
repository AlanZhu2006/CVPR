from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import IncrementalGaussianBuilder, load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert exported LingBot predictions into a Gaussian handle."
    )
    parser.add_argument("--predictions-npz", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument(
        "--dense-geometry-npz",
        default="",
        help="Optional normalized dense geometry exported by export_lingbot_dense_geometry.py.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submap-id", type=int, default=9000)
    parser.add_argument("--config", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    builder = IncrementalGaussianBuilder(
        output_dir=args.output_dir,
        config=config.memory,
    )
    if args.dense_geometry_npz:
        handle = builder.export_lingbot_dense_geometry_as_handle(
            submap_id=args.submap_id,
            dense_geometry_npz=args.dense_geometry_npz,
            reason="lingbot_dense_geometry_init",
        )
    else:
        if not args.predictions_npz or not args.summary_json:
            raise SystemExit(
                "Provide either --dense-geometry-npz or both --predictions-npz and --summary-json"
            )
        handle = builder.export_lingbot_predictions_as_handle(
            submap_id=args.submap_id,
            predictions_npz=args.predictions_npz,
            summary_json=args.summary_json,
            reason="lingbot_structured_init",
        )
    print(json.dumps(handle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
