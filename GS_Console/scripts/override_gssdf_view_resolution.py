#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Override GS-SDF view-mode render resolution by rewriting the saved scene config."
    )
    parser.add_argument("--scene-config", required=True, help="Path to model/config/scene/config.yaml")
    parser.add_argument("--width", type=int, help="Target render width")
    parser.add_argument("--height", type=int, help="Target render height")
    parser.add_argument("--scale", type=float, help="Uniform scale factor relative to the original config")
    parser.add_argument("--restore", action="store_true", help="Restore the original config from backup")
    return parser.parse_args()


def extract_value(text: str, key: str) -> float:
    match = re.search(rf"(^\s*{re.escape(key)}:\s*)([-+0-9.eE]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing `{key}` in scene config.")
    return float(match.group(2))


def replace_value(text: str, key: str, value: float | int) -> str:
    return re.sub(
        rf"(^\s*{re.escape(key)}:\s*)([-+0-9.eE]+)",
        rf"\g<1>{value}",
        text,
        count=1,
        flags=re.MULTILINE,
    )


def main() -> None:
    args = parse_args()
    scene_config = Path(args.scene_config).resolve()
    if not scene_config.is_file():
        raise SystemExit(f"Scene config does not exist: {scene_config}")

    backup_path = scene_config.with_name(f"{scene_config.name}.bak_view_res")
    current_text = scene_config.read_text(encoding="utf-8")

    if args.restore:
        if not backup_path.is_file():
            raise SystemExit(f"No resolution backup exists for {scene_config}")
        scene_config.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(
            json.dumps(
                {
                    "sceneConfig": str(scene_config),
                    "restoredFrom": str(backup_path),
                },
                indent=2,
            )
        )
        return

    if args.width is None and args.height is None and args.scale is None:
        raise SystemExit("Specify --width/--height or --scale, or pass --restore.")

    base_text = backup_path.read_text(encoding="utf-8") if backup_path.is_file() else current_text
    if not backup_path.is_file():
        backup_path.write_text(current_text, encoding="utf-8")

    original_width = int(round(extract_value(base_text, "width")))
    original_height = int(round(extract_value(base_text, "height")))
    original_fx = extract_value(base_text, "fx")
    original_fy = extract_value(base_text, "fy")
    original_cx = extract_value(base_text, "cx")
    original_cy = extract_value(base_text, "cy")

    if args.scale is not None:
        if args.scale <= 0:
            raise SystemExit("--scale must be positive.")
        target_width = max(1, int(round(original_width * args.scale)))
        target_height = max(1, int(round(original_height * args.scale)))
    else:
        target_width = args.width
        target_height = args.height
        if target_width is None and target_height is not None:
            target_width = max(1, int(round(original_width * (target_height / original_height))))
        if target_height is None and target_width is not None:
            target_height = max(1, int(round(original_height * (target_width / original_width))))
        if target_width is None or target_height is None:
            raise SystemExit("Failed to infer target width/height.")

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    updated_text = base_text
    updated_text = replace_value(updated_text, "width", int(target_width))
    updated_text = replace_value(updated_text, "height", int(target_height))
    updated_text = replace_value(updated_text, "fx", repr(original_fx * scale_x))
    updated_text = replace_value(updated_text, "fy", repr(original_fy * scale_y))
    updated_text = replace_value(updated_text, "cx", repr(original_cx * scale_x))
    updated_text = replace_value(updated_text, "cy", repr(original_cy * scale_y))

    scene_config.write_text(updated_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "sceneConfig": str(scene_config),
                "backupPath": str(backup_path),
                "original": {
                    "width": original_width,
                    "height": original_height,
                    "fx": original_fx,
                    "fy": original_fy,
                    "cx": original_cx,
                    "cy": original_cy,
                },
                "updated": {
                    "width": int(target_width),
                    "height": int(target_height),
                    "fx": original_fx * scale_x,
                    "fy": original_fy * scale_y,
                    "cx": original_cx * scale_x,
                    "cy": original_cy * scale_y,
                },
                "scale": {
                    "x": scale_x,
                    "y": scale_y,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
