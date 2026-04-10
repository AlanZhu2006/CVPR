from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from nuc_runtime.models import FramePacket


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STEREO_LAYOUT_CANDIDATES = (
    ("left", "right"),
    ("image_0", "image_1"),
    ("cam0/data", "cam1/data"),
    ("cam0", "cam1"),
)
ROSBAG_IMAGE_MSGTYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/Image",
    "sensor_msgs/CompressedImage",
}
ROSBAG_STEREO_TOPIC_CANDIDATES = (
    ("/cam0/image_raw", "/cam1/image_raw"),
    ("/cam0/image_raw/compressed", "/cam1/image_raw/compressed"),
    ("/stereo/left/image_raw", "/stereo/right/image_raw"),
    ("/stereo/left/image_raw/compressed", "/stereo/right/image_raw/compressed"),
    ("/camera/left/image_raw", "/camera/right/image_raw"),
    ("/camera/left/image_raw/compressed", "/camera/right/image_raw/compressed"),
)


def _resize_if_needed(frame, resize_width: int):
    if resize_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame
    scale = resize_width / float(width)
    new_height = int(height * scale)
    return cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)


def _iter_image_paths(directory: Path) -> list[Path]:
    return sorted(item for item in directory.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)


def _is_rosbag_path(path: Path) -> bool:
    return path.suffix.lower() == ".bag" or (path.is_dir() and (path / "metadata.yaml").is_file())


def _find_stereo_dirs(root: Path) -> tuple[Path, Path] | None:
    for left_rel, right_rel in STEREO_LAYOUT_CANDIDATES:
        left_dir = root / left_rel
        right_dir = root / right_rel
        if left_dir.is_dir() and right_dir.is_dir():
            return left_dir, right_dir
    return None


def _pair_stereo_images(left_paths: list[Path], right_paths: list[Path]) -> list[tuple[Path, Path]]:
    right_by_name = {item.name: item for item in right_paths}
    right_by_stem = {item.stem: item for item in right_paths}

    pairs: list[tuple[Path, Path]] = []
    for left_path in left_paths:
        right_path = right_by_name.get(left_path.name)
        if right_path is None:
            right_path = right_by_stem.get(left_path.stem)
        if right_path is not None:
            pairs.append((left_path, right_path))

    if pairs:
        return pairs

    pair_count = min(len(left_paths), len(right_paths))
    return list(zip(left_paths[:pair_count], right_paths[:pair_count]))


def _normalize_msgtype(msgtype: str) -> str:
    if "/msg/" in msgtype:
        return msgtype
    package, name = msgtype.split("/", 1)
    return f"{package}/msg/{name}"


def _rosbag_image_connections(reader) -> list:
    return [item for item in reader.connections if _normalize_msgtype(item.msgtype) in ROSBAG_IMAGE_MSGTYPES]


def _pick_rosbag_connections(reader, left_topic: str | None, right_topic: str | None) -> tuple[object, object | None]:
    image_connections = _rosbag_image_connections(reader)
    by_topic = {item.topic: item for item in image_connections}

    if left_topic:
        left_conn = by_topic.get(left_topic)
        if left_conn is None:
            available = ", ".join(sorted(by_topic))
            raise ValueError(f"Left image topic not found: {left_topic}. Available image topics: {available}")
        if right_topic:
            right_conn = by_topic.get(right_topic)
            if right_conn is None:
                available = ", ".join(sorted(by_topic))
                raise ValueError(f"Right image topic not found: {right_topic}. Available image topics: {available}")
            return left_conn, right_conn
        return left_conn, None

    topic_names = set(by_topic)
    for candidate_left, candidate_right in ROSBAG_STEREO_TOPIC_CANDIDATES:
        if candidate_left in topic_names and candidate_right in topic_names:
            return by_topic[candidate_left], by_topic[candidate_right]

    left_candidates = sorted(item for item in topic_names if "left" in item)
    right_candidates = sorted(item for item in topic_names if "right" in item)
    if left_candidates and right_candidates:
        return by_topic[left_candidates[0]], by_topic[right_candidates[0]]

    cam0_candidates = sorted(item for item in topic_names if "/cam0/" in item or item.endswith("/cam0"))
    cam1_candidates = sorted(item for item in topic_names if "/cam1/" in item or item.endswith("/cam1"))
    if cam0_candidates and cam1_candidates:
        return by_topic[cam0_candidates[0]], by_topic[cam1_candidates[0]]

    if len(image_connections) == 1:
        return image_connections[0], None

    available = ", ".join(sorted(by_topic))
    raise ValueError(
        "Could not auto-detect image topics from rosbag. "
        f"Please set rosbag_left_topic/rosbag_right_topic. Available image topics: {available}"
    )


def _message_data_to_uint8(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.uint8, copy=False).reshape(-1)
    return np.frombuffer(bytes(data), dtype=np.uint8)


def _decode_rosbag_image(message, msgtype: str) -> np.ndarray:
    normalized = _normalize_msgtype(msgtype)
    if normalized.endswith("CompressedImage"):
        buffer = _message_data_to_uint8(message.data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode compressed image from rosbag.")
        return frame

    encoding = getattr(message, "encoding", "").lower()
    height = int(message.height)
    width = int(message.width)
    data = _message_data_to_uint8(message.data)

    if encoding in {"mono8", "8uc1"}:
        gray = data.reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if encoding in {"bgr8", "8uc3"}:
        return data.reshape(height, width, 3)

    if encoding == "rgb8":
        rgb = data.reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if encoding == "bgra8":
        bgra = data.reshape(height, width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    if encoding == "rgba8":
        rgba = data.reshape(height, width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    if encoding in {"mono16", "16uc1"}:
        gray16 = np.frombuffer(bytes(message.data), dtype=np.uint16).reshape(height, width)
        gray8 = cv2.convertScaleAbs(gray16, alpha=255.0 / max(float(gray16.max()), 1.0))
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported rosbag image encoding: {message.encoding}")


def _iter_rosbag_frames(
    input_path: Path,
    frame_step: int,
    max_frames: int,
    resize_width: int,
    left_topic: str | None,
    right_topic: str | None,
    sync_tolerance_sec: float,
) -> Iterator[FramePacket]:
    try:
        from rosbags.highlevel import AnyReader
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("rosbags is required to read rosbag inputs. Install it with `pip install rosbags`.") from exc

    tolerance_ns = int(max(sync_tolerance_sec, 0.0) * 1e9)
    emitted = 0
    pair_idx = 0

    with AnyReader([input_path]) as reader:
        left_conn, right_conn = _pick_rosbag_connections(reader, left_topic, right_topic)
        connections = [left_conn] if right_conn is None else [left_conn, right_conn]

        pending_left: tuple[int, np.ndarray] | None = None
        pending_right: tuple[int, np.ndarray] | None = None

        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            frame = _resize_if_needed(_decode_rosbag_image(message, connection.msgtype), resize_width)

            if right_conn is None:
                if pair_idx % frame_step == 0:
                    yield FramePacket(
                        frame_idx=emitted,
                        timestamp_sec=timestamp_ns / 1e9,
                        frame_bgr=frame,
                        source_name=f"{connection.topic}:{timestamp_ns}",
                    )
                    emitted += 1
                    if max_frames > 0 and emitted >= max_frames:
                        break
                pair_idx += 1
                continue

            if connection.topic == left_conn.topic:
                pending_left = (timestamp_ns, frame)
            elif connection.topic == right_conn.topic:
                pending_right = (timestamp_ns, frame)

            while pending_left is not None and pending_right is not None:
                delta_ns = pending_left[0] - pending_right[0]
                if abs(delta_ns) <= tolerance_ns:
                    if pair_idx % frame_step == 0:
                        timestamp_sec = max(pending_left[0], pending_right[0]) / 1e9
                        yield FramePacket(
                            frame_idx=emitted,
                            timestamp_sec=timestamp_sec,
                            frame_bgr=pending_left[1],
                            source_name=f"{left_conn.topic}:{pending_left[0]}",
                            frame_bgr_right=pending_right[1],
                            right_source_name=f"{right_conn.topic}:{pending_right[0]}",
                        )
                        emitted += 1
                        if max_frames > 0 and emitted >= max_frames:
                            return
                    pair_idx += 1
                    pending_left = None
                    pending_right = None
                    continue

                if delta_ns < 0:
                    pending_left = None
                else:
                    pending_right = None


def iter_frames(
    input_path: str | Path,
    frame_step: int = 1,
    max_frames: int = 0,
    default_fps: float = 30.0,
    resize_width: int = 0,
    rosbag_left_topic: str | None = None,
    rosbag_right_topic: str | None = None,
    rosbag_sync_tolerance_sec: float = 0.01,
) -> Iterator[FramePacket]:
    path = Path(input_path)
    frame_step = max(frame_step, 1)

    if _is_rosbag_path(path):
        yield from _iter_rosbag_frames(
            input_path=path,
            frame_step=frame_step,
            max_frames=max_frames,
            resize_width=resize_width,
            left_topic=rosbag_left_topic,
            right_topic=rosbag_right_topic,
            sync_tolerance_sec=rosbag_sync_tolerance_sec,
        )
        return

    if path.is_dir():
        stereo_dirs = _find_stereo_dirs(path)
        if stereo_dirs is not None:
            left_dir, right_dir = stereo_dirs
            stereo_pairs = _pair_stereo_images(_iter_image_paths(left_dir), _iter_image_paths(right_dir))
            if max_frames > 0:
                stereo_pairs = stereo_pairs[: max_frames * frame_step]

            emitted = 0
            for idx, (left_path, right_path) in enumerate(stereo_pairs[::frame_step]):
                left_frame = cv2.imread(str(left_path))
                right_frame = cv2.imread(str(right_path))
                if left_frame is None or right_frame is None:
                    continue
                left_frame = _resize_if_needed(left_frame, resize_width)
                right_frame = _resize_if_needed(right_frame, resize_width)
                yield FramePacket(
                    frame_idx=idx,
                    timestamp_sec=idx / float(default_fps),
                    frame_bgr=left_frame,
                    source_name=left_path.name,
                    frame_bgr_right=right_frame,
                    right_source_name=right_path.name,
                )
                emitted += 1
                if max_frames > 0 and emitted >= max_frames:
                    break
            return

        image_paths = _iter_image_paths(path)
        if max_frames > 0:
            image_paths = image_paths[: max_frames * frame_step]

        emitted = 0
        for idx, image_path in enumerate(image_paths[::frame_step]):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            frame = _resize_if_needed(frame, resize_width)
            yield FramePacket(
                frame_idx=idx,
                timestamp_sec=idx / float(default_fps),
                frame_bgr=frame,
                source_name=image_path.name,
            )
            emitted += 1
            if max_frames > 0 and emitted >= max_frames:
                break
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input path: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = default_fps

    frame_idx = 0
    emitted = 0
    source_name = path.name
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        frame = _resize_if_needed(frame, resize_width)
        yield FramePacket(
            frame_idx=emitted,
            timestamp_sec=frame_idx / float(fps),
            frame_bgr=frame,
            source_name=f"{source_name}:{frame_idx}",
        )
        emitted += 1
        frame_idx += 1
        if max_frames > 0 and emitted >= max_frames:
            break

    cap.release()
