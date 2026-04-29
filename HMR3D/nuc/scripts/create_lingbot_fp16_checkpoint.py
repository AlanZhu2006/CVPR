from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smaller LingBot checkpoint for embedded experiments."
    )
    parser.add_argument("--input", required=True, help="Input LingBot .pt checkpoint.")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint.")
    parser.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="State-dict prefix to drop, for example camera_head or point_head.",
    )
    parser.add_argument("--fp16", action="store_true", help="Convert floating tensors to fp16.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drop_prefixes = tuple(prefix.rstrip(".") + "." for prefix in args.drop_prefix)

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False, mmap=True)
    source = ckpt.get("model", ckpt)
    if not isinstance(source, dict):
        raise TypeError("Checkpoint does not contain a state dict or {'model': state_dict}.")

    converted = {}
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "drop_prefixes": list(args.drop_prefix),
        "fp16": bool(args.fp16),
        "kept_tensors": 0,
        "dropped_tensors": 0,
        "kept_numel": 0,
        "dropped_numel": 0,
    }
    for key, value in source.items():
        if drop_prefixes and key.startswith(drop_prefixes):
            if torch.is_tensor(value):
                stats["dropped_tensors"] += 1
                stats["dropped_numel"] += int(value.numel())
            continue
        if torch.is_tensor(value):
            stats["kept_tensors"] += 1
            stats["kept_numel"] += int(value.numel())
            if args.fp16 and torch.is_floating_point(value):
                value = value.to(torch.float16)
        converted[key] = value

    torch.save({"model": converted, "embedded_export": stats}, output_path)
    stats["output_size_gib"] = output_path.stat().st_size / (1024**3)
    print(stats)


if __name__ == "__main__":
    main()
