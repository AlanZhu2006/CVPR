from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
LINGBOT_ROOT = REPO_ROOT / "third_party_research" / "lingbot-map"
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.vis.point_cloud_viewer import PointCloudViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open exported LingBot predictions in the official viser point cloud viewer."
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--point-size", type=float, default=1e-5)
    parser.add_argument("--init-conf-threshold", type=float, default=1.0)
    parser.add_argument("--use-point-map", action="store_true", default=False)
    parser.add_argument("--mask-sky", action="store_true", default=False)
    parser.add_argument("--depth-stride", type=int, default=1)
    return parser.parse_args()


def infer_image_folder(image_paths: list[str]) -> str | None:
    if not image_paths:
        return None
    parents = {str(Path(path).expanduser().resolve().parent) for path in image_paths}
    if len(parents) == 1:
        return next(iter(parents))
    return None


def main() -> None:
    args = parse_args()
    predictions_npz = Path(args.predictions_npz).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()
    if not predictions_npz.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_npz}")
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_json}")

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    image_paths = [str(Path(path).expanduser().resolve()) for path in summary.get("image_paths", [])]
    if not image_paths:
        raise RuntimeError(f"No image paths stored in {summary_json}")

    predictions = {key: value for key, value in np.load(predictions_npz).items()}
    images = load_and_preprocess_images(
        image_paths,
        mode="crop",
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    predictions["images"] = images.detach().cpu().numpy()

    image_folder = infer_image_folder(image_paths)
    viewer = PointCloudViewer(
        pred_dict=predictions,
        port=args.port,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        init_conf_threshold=args.init_conf_threshold,
        use_point_map=args.use_point_map,
        mask_sky=args.mask_sky,
        image_folder=image_folder,
        depth_stride=args.depth_stride,
    )
    viewer.downsample_slider.value = max(1, int(args.downsample_factor))
    viewer.psize_slider.value = float(args.point_size)
    viewer.vis_threshold_slider.value = float(np.clip(args.init_conf_threshold, 1.0, 5.0))
    print(f"Starting LingBot viser viewer on http://127.0.0.1:{args.port}")
    viewer.run()


if __name__ == "__main__":
    main()
