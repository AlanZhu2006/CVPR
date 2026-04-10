from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nuc_runtime.config import OutputConfig
from nuc_runtime.models import EventRecord, TrackingOutput


class ArtifactWriter:
    def __init__(self, output_dir: str | Path, config: OutputConfig):
        self.root = Path(output_dir)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self.keyframe_dir = self.root / "keyframes"
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.root / "events.jsonl"
        self.summary_path = self.root / "summary.json"
        self.submaps_path = self.root / "submaps.json"
        self.poses_path = self.root / "poses.csv"

        self.pose_file = self.poses_path.open("w", newline="", encoding="utf-8")
        self.pose_writer = csv.DictWriter(
            self.pose_file,
            fieldnames=[
                "frame_idx",
                "timestamp_sec",
                "is_keyframe",
                "track_ok",
                "match_count",
                "inlier_count",
                "pixel_motion",
                "tracking_mode",
                "stereo_points",
                "tx",
                "ty",
                "tz",
                "active_id",
                "bank_size",
                "recoveries",
            ],
        )
        self.pose_writer.writeheader()
        self.video_writer = None

    def save_keyframe_image(self, frame_idx: int, frame_bgr: np.ndarray) -> str | None:
        if not self.config.save_keyframe_images:
            return None
        path = self.keyframe_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(path), frame_bgr)
        return str(path)

    def append_pose(self, tracking: TrackingOutput, snapshot: dict[str, Any]) -> None:
        tx, ty, tz = tracking.pose[:3, 3].tolist()
        self.pose_writer.writerow(
            {
                "frame_idx": tracking.frame_idx,
                "timestamp_sec": f"{tracking.timestamp_sec:.6f}",
                "is_keyframe": int(tracking.is_keyframe),
                "track_ok": int(tracking.track_ok),
                "match_count": tracking.match_count,
                "inlier_count": tracking.inlier_count,
                "pixel_motion": f"{tracking.pixel_motion:.3f}",
                "tracking_mode": tracking.notes.get("tracking_mode"),
                "stereo_points": tracking.notes.get("stereo_points", 0),
                "tx": f"{tx:.6f}",
                "ty": f"{ty:.6f}",
                "tz": f"{tz:.6f}",
                "active_id": snapshot.get("active_id"),
                "bank_size": snapshot.get("bank_size"),
                "recoveries": snapshot.get("recoveries"),
            }
        )

    def append_events(self, events: list[EventRecord]) -> None:
        if not events:
            return
        with self.events_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(
                    json.dumps(
                        {
                            "frame_idx": event.frame_idx,
                            "timestamp_sec": event.timestamp_sec,
                            "event_type": event.event_type,
                            "payload": event.payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def maybe_write_debug_frame(
        self,
        frame_bgr: np.ndarray,
        tracking: TrackingOutput,
        snapshot: dict[str, Any],
        last_event: str | None,
    ) -> None:
        if not self.config.save_debug_video:
            return

        canvas = frame_bgr.copy()
        lines = [
            f"frame={tracking.frame_idx} keyframe={int(tracking.is_keyframe)} track_ok={int(tracking.track_ok)}",
            f"matches={tracking.match_count} inliers={tracking.inlier_count} motion={tracking.pixel_motion:.2f}",
            f"mode={tracking.notes.get('tracking_mode', '-')} stereo_pts={tracking.notes.get('stereo_points', 0)}",
            f"active={snapshot.get('active_id')} active_kf={snapshot.get('active_keyframes')} bank={snapshot.get('bank_size')}",
            f"recoveries={snapshot.get('recoveries')} last_event={last_event or '-'}",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (20, 30 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.video_writer is None:
            height, width = canvas.shape[:2]
            path = self.root / self.config.debug_video_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(path),
                fourcc,
                self.config.debug_video_fps,
                (width, height),
            )

        if self.video_writer is not None:
            self.video_writer.write(canvas)

    def write_summary(self, summary: dict[str, Any], submaps: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.submaps_path.write_text(
            json.dumps(submaps, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.pose_file.close()
        if self.video_writer is not None:
            self.video_writer.release()
