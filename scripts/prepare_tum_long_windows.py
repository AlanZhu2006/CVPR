#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


def read_file_list(path: Path) -> dict[float, list[str]]:
    lines = path.read_text(encoding="utf-8").replace(",", " ").replace("\t", " ").splitlines()
    parsed = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = [value.strip() for value in line.split(" ") if value.strip()]
        if len(fields) <= 1:
            continue
        parsed.append((float(fields[0]), fields[1:]))
    return dict(parsed)


def associate(
    first_list: dict[float, list[str]],
    second_list: dict[float, list[str]],
    offset: float = 0.0,
    max_difference: float = 0.02,
) -> list[tuple[float, float]]:
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches: list[tuple[float, float]] = []
    for _, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


@dataclass(frozen=True)
class WindowSpec:
    source_seq: str
    window_name: str
    start: int
    length: int

    @property
    def output_seq(self) -> str:
        return f"{self.source_seq}_{self.window_name}"


DEFAULT_WINDOWS = [
    WindowSpec("rgbd_dataset_freiburg3_sitting_static", "head500", 0, 500),
    WindowSpec("rgbd_dataset_freiburg3_sitting_static", "tail500", 203, 500),
    WindowSpec("rgbd_dataset_freiburg3_sitting_xyz", "head500", 0, 500),
    WindowSpec("rgbd_dataset_freiburg3_sitting_xyz", "mid500", 250, 500),
    WindowSpec("rgbd_dataset_freiburg3_sitting_xyz", "tail500", 758, 500),
]


def materialize_window(raw_root: Path, output_root: Path, spec: WindowSpec) -> None:
    source_dir = raw_root / spec.source_seq
    rgb_list = read_file_list(source_dir / "rgb.txt")
    gt_list = read_file_list(source_dir / "groundtruth.txt")
    matches = associate(rgb_list, gt_list)
    if spec.start + spec.length > len(matches):
        raise ValueError(
            f"Window {spec.output_seq} exceeds matched trajectory length {len(matches)}."
        )

    selected = matches[spec.start : spec.start + spec.length]
    output_seq_dir = output_root / spec.output_seq
    output_rgb_dir = output_seq_dir / "rgb_1000"
    if output_seq_dir.exists():
        shutil.rmtree(output_seq_dir)
    output_rgb_dir.mkdir(parents=True, exist_ok=True)

    gt_lines = []
    for rgb_stamp, gt_stamp in selected:
        rgb_rel = rgb_list[rgb_stamp][0]
        src_rgb = source_dir / rgb_rel
        dst_rgb = output_rgb_dir / Path(rgb_rel).name
        shutil.copy2(src_rgb, dst_rgb)
        gt_lines.append(" ".join([str(gt_stamp), *gt_list[gt_stamp]]))

    (output_seq_dir / "groundtruth_1000.txt").write_text(
        "\n".join(gt_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw/tum")
    parser.add_argument("--output-root", default="data/long_tum_s1")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for spec in DEFAULT_WINDOWS:
        materialize_window(raw_root=raw_root, output_root=output_root, spec=spec)
        print(spec.output_seq)


if __name__ == "__main__":
    main()
