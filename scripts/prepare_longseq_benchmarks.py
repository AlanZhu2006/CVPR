#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def _copy_frames(src_paths, dst_dir: Path, target_frames: int, stride: int = 1, formatter: str = "frame_{:04d}"):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    selected = src_paths[: target_frames * stride : stride]
    for idx, src in enumerate(selected):
        dst = dst_dir / formatter.format(idx)
        if "." not in dst.name:
            dst = dst.with_suffix(Path(src).suffix)
        shutil.copy(src, dst)
    return len(selected)


def prepare_scannet(raw_root: Path, target_frames: int, stride: int = 3) -> None:
    scans_root = raw_root / "scans_test"
    out_root = DATA_ROOT / f"long_scannet_s{stride}"
    for seq in sorted(p.name for p in scans_root.iterdir() if p.is_dir()):
        img_paths = sorted((scans_root / seq / "color").glob("*.jpg"), key=lambda p: int(p.stem))
        depth_paths = sorted((scans_root / seq / "depth").glob("*.png"), key=lambda p: int(p.stem))
        pose_paths = sorted((scans_root / seq / "pose").glob("*.txt"), key=lambda p: int(p.stem))
        actual_frames = min(target_frames, min(len(img_paths), len(depth_paths), len(pose_paths)) // stride)
        color_dir = out_root / seq / f"color_{target_frames}"
        depth_dir = out_root / seq / f"depth_{target_frames}"
        _copy_frames([str(p) for p in img_paths], color_dir, actual_frames, stride=stride)
        _copy_frames([str(p) for p in depth_paths], depth_dir, actual_frames, stride=stride)
        pose_out = out_root / seq / f"pose_{target_frames}.txt"
        pose_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pose_out, "w", encoding="utf-8") as handle:
            for pose_path in pose_paths[: actual_frames * stride : stride]:
                pose = np.loadtxt(pose_path).reshape(-1)
                handle.write(" ".join(map(str, pose)) + "\n")


def read_tum_file_list(filename: Path):
    lines = filename.read_text(encoding="utf-8").replace(",", " ").replace("\t", " ").splitlines()
    items = [[token.strip() for token in line.split(" ") if token.strip()] for line in lines if line and not line.startswith("#")]
    return {float(line[0]): line[1:] for line in items if len(line) > 1}


def associate_tum(first, second, offset=0.0, max_difference=0.02):
    first_keys = set(first.keys())
    second_keys = set(second.keys())
    candidates = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    candidates.sort()
    matches = []
    for _, a, b in candidates:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


def prepare_tum(raw_root: Path, target_frames: int, stride: int = 1) -> None:
    out_root = DATA_ROOT / f"long_tum_s{stride}"
    for sequence_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        rgb = read_tum_file_list(sequence_dir / "rgb.txt")
        gt = read_tum_file_list(sequence_dir / "groundtruth.txt")
        matches = associate_tum(rgb, gt)
        frames = [sequence_dir / rgb[a][0] for a, _ in matches]
        poses = [[b] + gt[b] for _, b in matches]
        seq_name = sequence_dir.name
        rgb_dir = out_root / seq_name / f"rgb_{target_frames}"
        copied = _copy_frames([str(frame) for frame in frames], rgb_dir, target_frames, stride=stride, formatter="{}.png")
        pose_out = out_root / seq_name / f"groundtruth_{target_frames}.txt"
        pose_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pose_out, "w", encoding="utf-8") as handle:
            for pose in poses[: copied * stride : stride]:
                handle.write(" ".join(map(str, pose)) + "\n")


def prepare_bonn(raw_root: Path, target_frames: int, start_frame: int = 30) -> None:
    out_root = DATA_ROOT / "long_bonn_s1" / "rgbd_bonn_dataset"
    for sequence_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        rgb_frames = sorted((sequence_dir / "rgb").glob("*.png"))
        depth_frames = sorted((sequence_dir / "depth").glob("*.png"))
        gt = np.loadtxt(sequence_dir / "groundtruth.txt")
        actual_count = min(len(rgb_frames), len(depth_frames), len(gt)) - start_frame
        actual_count = min(actual_count, target_frames)
        if actual_count <= 0:
            continue
        seq_name = sequence_dir.name
        rgb_dir = out_root / seq_name / f"rgb_{target_frames}"
        depth_dir = out_root / seq_name / f"depth_{target_frames}"
        _copy_frames([str(path) for path in rgb_frames[start_frame : start_frame + actual_count]], rgb_dir, actual_count, formatter="{}.png")
        _copy_frames([str(path) for path in depth_frames[start_frame : start_frame + actual_count]], depth_dir, actual_count, formatter="{}.png")
        np.savetxt(out_root / seq_name / f"groundtruth_{target_frames}.txt", gt[start_frame : start_frame + actual_count])


def prepare_kitti(raw_root: Path, target_frames: int) -> None:
    out_root = DATA_ROOT / "long_kitti_s1" / "depth_selection" / "val_selection_cropped"
    depth_dirs = sorted(raw_root.glob("val/*/proj_depth/groundtruth/image_02"))
    for depth_dir in depth_dirs:
        seq_name = f"{depth_dir.parts[-4]}_02"
        depth_out = out_root / f"groundtruth_depth_gathered_{target_frames}" / seq_name
        image_out = out_root / f"image_gathered_{target_frames}" / seq_name
        depth_files = sorted(depth_dir.glob("*.png"))[:target_frames]
        if depth_out.exists():
            shutil.rmtree(depth_out)
        if image_out.exists():
            shutil.rmtree(image_out)
        depth_out.mkdir(parents=True, exist_ok=True)
        image_out.mkdir(parents=True, exist_ok=True)
        for depth_file in depth_files:
            shutil.copy(depth_file, depth_out / depth_file.name)
            prefix = "_".join(depth_file.parts[-5].split("_")[:3])
            image_file = Path(str(depth_file).replace("val", prefix).replace("proj_depth/groundtruth/image_02", "image_02/data"))
            if image_file.exists():
                shutil.copy(image_file, image_out / image_file.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scannet", "tum", "bonn", "kitti", "all"], default="all")
    parser.add_argument("--target-frames", type=int, default=None)
    parser.add_argument("--scannet-root", type=Path)
    parser.add_argument("--tum-root", type=Path)
    parser.add_argument("--bonn-root", type=Path)
    parser.add_argument("--kitti-root", type=Path)
    args = parser.parse_args()

    if args.dataset in {"scannet", "all"}:
        if args.scannet_root is None:
            print("Skipping ScanNet: --scannet-root was not provided.")
        else:
            for frames in ([args.target_frames] if args.target_frames else [1000]):
                prepare_scannet(args.scannet_root, frames)

    if args.dataset in {"tum", "all"}:
        if args.tum_root is None:
            print("Skipping TUM: --tum-root was not provided.")
        else:
            for frames in ([args.target_frames] if args.target_frames else [1000]):
                prepare_tum(args.tum_root, frames)

    if args.dataset in {"bonn", "all"}:
        if args.bonn_root is None:
            print("Skipping Bonn: --bonn-root was not provided.")
        else:
            for frames in ([args.target_frames] if args.target_frames else [500]):
                prepare_bonn(args.bonn_root, frames)

    if args.dataset in {"kitti", "all"}:
        if args.kitti_root is None:
            print("Skipping KITTI: --kitti-root was not provided.")
        else:
            for frames in ([args.target_frames] if args.target_frames else [500]):
                prepare_kitti(args.kitti_root, frames)


if __name__ == "__main__":
    main()

