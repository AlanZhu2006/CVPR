from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a FAST-LIVO NTU VIRAL config with pose_output_en enabled."
    )
    parser.add_argument(
        "--input-yaml",
        default="/home/nyu/Codespace/CVPR/third_party_research/FAST-LIVO/config/NTU_VIRAL.yaml",
        help="Input FAST-LIVO NTU VIRAL yaml.",
    )
    parser.add_argument("--output-yaml", required=True, help="Output yaml path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_yaml = Path(args.input_yaml).expanduser().resolve()
    output_yaml = Path(args.output_yaml).expanduser().resolve()
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(input_yaml.read_text(encoding="utf-8"))
    data["pose_output_en"] = True
    output_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(output_yaml)


if __name__ == "__main__":
    main()
