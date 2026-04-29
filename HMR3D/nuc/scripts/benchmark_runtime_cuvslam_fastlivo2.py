from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import psutil
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark runtime and Jetson resource usage for cuVSLAM and FAST-LIVO2 on NTU VIRAL."
    )
    parser.add_argument("--output-dir", required=True, help="Directory to store runtime benchmark artifacts.")
    parser.add_argument(
        "--sequence-dir",
        default="/home/nyu/Codespace/CVPR/nuc_output/ntuviral_spms02_cuvslam_input",
        help="Extracted NTU VIRAL image directory for cuVSLAM.",
    )
    parser.add_argument(
        "--camera-yaml",
        default="/home/nyu/Codespace/CVPR/third_party_research/FAST-LIVO2/config/camera_NTU_VIRAL.yaml",
        help="Camera YAML for cuVSLAM.",
    )
    parser.add_argument(
        "--cuvslam-script",
        default="/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_ntuviral.py",
        help="Path to cuVSLAM NTU VIRAL runner.",
    )
    parser.add_argument(
        "--cuvslam-python",
        default="/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python",
        help="Python executable with cuvslam bindings installed.",
    )
    parser.add_argument(
        "--fastlivo-script",
        default="/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_fastlivo2_ntuviral_full.sh",
        help="Path to FAST-LIVO2 NTU VIRAL runner.",
    )
    parser.add_argument(
        "--fastlivo-bag-metadata",
        default="/home/nyu/Codespace/CVPR/datasets/ntu_viral/spms_02_fastlivo2_ros2/metadata.yaml",
        help="ROS2 bag metadata.yaml for FAST-LIVO2 input duration/count.",
    )
    parser.add_argument(
        "--tegrastats-interval-ms",
        type=int,
        default=1000,
        help="Sampling interval for tegrastats.",
    )
    return parser.parse_args()


def load_sequence_duration_and_frames(sequence_dir: Path) -> tuple[float, int]:
    times = [float(line.strip()) for line in (sequence_dir / "times.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    return times[-1] - times[0], len(times)


def load_bag_duration_and_counts(metadata_path: Path) -> tuple[float, int, int, int]:
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = metadata["rosbag2_bagfile_information"]
    duration_sec = info["duration"]["nanoseconds"] * 1e-9
    image_count = 0
    lidar_count = 0
    imu_count = 0
    for topic in info["topics_with_message_count"]:
        name = topic["topic_metadata"]["name"]
        count = int(topic["message_count"])
        if name == "/left/image_raw":
            image_count = count
        elif name == "/os1_cloud_node1/points":
            lidar_count = count
        elif name == "/imu/imu":
            imu_count = count
    return duration_sec, image_count, lidar_count, imu_count


def parse_tegrastats_log(path: Path) -> dict:
    ram_used = []
    cpu_totals = []
    gr3d = []
    vdd_in = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ram_match = re.search(r"RAM\s+([0-9]+)\/([0-9]+)MB", line)
        if ram_match:
            ram_used.append(float(ram_match.group(1)))
        cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if cpu_match:
            cpu_entries = cpu_match.group(1).split(",")
            total = 0.0
            count = 0
            for entry in cpu_entries:
                load_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", entry)
                if load_match:
                    total += float(load_match.group(1))
                    count += 1
            if count > 0:
                cpu_totals.append(total / count)
        gr3d_match = re.search(r"GR3D_FREQ\s+([0-9]+(?:\.[0-9]+)?)%", line)
        if gr3d_match:
            gr3d.append(float(gr3d_match.group(1)))
        vdd_match = re.search(r"VDD_IN\s+([0-9]+)mW", line)
        if vdd_match:
            vdd_in.append(float(vdd_match.group(1)))
    return {
        "samples": len(ram_used),
        "avg_ram_mb": float(sum(ram_used) / len(ram_used)) if ram_used else None,
        "max_ram_mb": float(max(ram_used)) if ram_used else None,
        "avg_cpu_percent": float(sum(cpu_totals) / len(cpu_totals)) if cpu_totals else None,
        "max_cpu_percent": float(max(cpu_totals)) if cpu_totals else None,
        "avg_gr3d_percent": float(sum(gr3d) / len(gr3d)) if gr3d else None,
        "max_gr3d_percent": float(max(gr3d)) if gr3d else None,
        "avg_vdd_in_mw": float(sum(vdd_in) / len(vdd_in)) if vdd_in else None,
        "max_vdd_in_mw": float(max(vdd_in)) if vdd_in else None,
    }


def run_monitored_command(command: str, output_dir: Path, label: str, env: dict[str, str] | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tegra_log = output_dir / f"{label}_tegrastats.log"
    stdout_log = output_dir / f"{label}_stdout.log"
    stderr_log = output_dir / f"{label}_stderr.log"

    tegra_file = tegra_log.open("w", encoding="utf-8")
    tegra_proc = subprocess.Popen(
        ["/usr/bin/tegrastats", "--interval", str(args.tegrastats_interval_ms)],
        stdout=tegra_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.monotonic()
    try:
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                cwd="/home/nyu/Codespace/CVPR",
                env=full_env,
                stdout=out,
                stderr=err,
                text=True,
            )
            ps_proc = psutil.Process(proc.pid)
            max_rss_kb = 0.0
            cpu_user_sec = 0.0
            cpu_system_sec = 0.0
            while proc.poll() is None:
                try:
                    procs = [ps_proc] + ps_proc.children(recursive=True)
                except psutil.Error:
                    procs = []
                rss_bytes = 0
                user_sec = 0.0
                sys_sec = 0.0
                for p in procs:
                    try:
                        mi = p.memory_info()
                        ct = p.cpu_times()
                    except psutil.Error:
                        continue
                    rss_bytes += getattr(mi, "rss", 0)
                    user_sec += getattr(ct, "user", 0.0)
                    sys_sec += getattr(ct, "system", 0.0)
                max_rss_kb = max(max_rss_kb, rss_bytes / 1024.0)
                cpu_user_sec = max(cpu_user_sec, user_sec)
                cpu_system_sec = max(cpu_system_sec, sys_sec)
                time.sleep(max(args.tegrastats_interval_ms / 1000.0, 0.2))
            proc.wait()
    finally:
        tegra_proc.send_signal(signal.SIGINT)
        try:
            tegra_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tegra_proc.kill()
        tegra_file.close()
    wall_time_sec = time.monotonic() - start
    tegra_metrics = parse_tegrastats_log(tegra_log)

    metrics = {
        "returncode": int(proc.returncode),
        "wall_time_sec": float(wall_time_sec),
        "user_time_sec": float(cpu_user_sec),
        "system_time_sec": float(cpu_system_sec),
        "max_rss_kb": float(max_rss_kb),
        **tegra_metrics,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "tegrastats_log": str(tegra_log),
    }
    return metrics


def summarize_cuvslam(output_dir: Path, seq_duration_sec: float, frame_count: int) -> dict:
    summary_path = output_dir / "cuvslam_ntuviral_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"cuVSLAM run did not produce summary file: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tracked = int(summary["tracked_frames"])
    return {
        "tracked_frames": tracked,
        "failed_frames": int(summary["failed_frames"]),
        "completion_ratio": float(tracked / max(frame_count, 1)),
        "input_frames": int(frame_count),
        "sequence_duration_sec": float(seq_duration_sec),
    }


def summarize_fastlivo(output_dir: Path, bag_duration_sec: float, image_count: int, lidar_count: int, imu_count: int) -> dict:
    traj_path = output_dir / "fastlivo2_tum.txt"
    if not traj_path.exists():
        raise RuntimeError(f"FAST-LIVO2 run did not produce trajectory file: {traj_path}")
    pose_lines = sum(1 for _ in traj_path.open("r", encoding="utf-8")) if traj_path.exists() else 0
    return {
        "output_poses": int(pose_lines),
        "input_image_frames": int(image_count),
        "input_lidar_scans": int(lidar_count),
        "input_imu_msgs": int(imu_count),
        "sequence_duration_sec": float(bag_duration_sec),
    }


def add_runtime_derived_fields(run_metrics: dict, seq_duration_sec: float, item_count: int) -> None:
    wall = run_metrics["wall_time_sec"]
    run_metrics["throughput_hz"] = float(item_count / wall) if wall > 1e-9 else None
    run_metrics["realtime_factor"] = float(seq_duration_sec / wall) if wall > 1e-9 else None
    if run_metrics.get("max_rss_kb") is not None:
        run_metrics["max_rss_mb"] = float(run_metrics["max_rss_kb"] / 1024.0)


def main() -> None:
    global args
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    seq_duration_sec, frame_count = load_sequence_duration_and_frames(sequence_dir)
    bag_duration_sec, image_count, lidar_count, imu_count = load_bag_duration_and_counts(
        Path(args.fastlivo_bag_metadata).expanduser().resolve()
    )

    cu_outdir = output_dir / "cuvslam_run"
    cu_traj = cu_outdir / "cuvslam_tum.txt"
    cu_python = Path(args.cuvslam_python).expanduser()
    cu_command = (
        f"{cu_python} {Path(args.cuvslam_script).expanduser().resolve()} "
        f"--sequence-dir {sequence_dir} "
        f"--camera-yaml {Path(args.camera_yaml).expanduser().resolve()} "
        f"--output-trajectory {cu_traj}"
    )
    cu_metrics = run_monitored_command(cu_command, cu_outdir, "cuvslam")
    if cu_metrics["returncode"] != 0:
        raise RuntimeError(
            "cuVSLAM runtime benchmark command failed. "
            f"See {cu_metrics['stderr_log']} and {cu_metrics['stdout_log']}."
        )
    cu_metrics.update(summarize_cuvslam(cu_outdir, seq_duration_sec, frame_count))
    add_runtime_derived_fields(cu_metrics, seq_duration_sec, frame_count)

    fl_outdir = output_dir / "fastlivo2_run"
    fl_command = str(Path(args.fastlivo_script).expanduser().resolve())
    fl_metrics = run_monitored_command(fl_command, fl_outdir, "fastlivo2")
    if fl_metrics["returncode"] != 0:
        raise RuntimeError(
            "FAST-LIVO2 runtime benchmark command failed. "
            f"See {fl_metrics['stderr_log']} and {fl_metrics['stdout_log']}."
        )
    fl_metrics.update(summarize_fastlivo(
        Path("/home/nyu/Codespace/CVPR/nuc_output/ntuviral_spms02_fastlivo2"),
        bag_duration_sec,
        image_count,
        lidar_count,
        imu_count,
    ))
    add_runtime_derived_fields(fl_metrics, bag_duration_sec, lidar_count)

    summary = {
        "dataset": "NTU VIRAL",
        "sequence": "spms_02",
        "cuvslam": cu_metrics,
        "fastlivo2": fl_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
