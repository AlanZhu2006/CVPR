from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a prepared LingBot dense job, run it on a remote GPU host, "
            "and download LingBot predictions."
        )
    )
    parser.add_argument("--remote", required=True, help="SSH target, for example user@gpu-host.")
    parser.add_argument("--job-dir", required=True, help="Prepared job directory.")
    parser.add_argument("--remote-root", required=True, help="Remote directory for uploaded jobs.")
    parser.add_argument("--remote-repo-root", required=True, help="Remote CVPR repo root.")
    parser.add_argument("--model-path", required=True, help="LingBot model path on the remote host.")
    parser.add_argument("--remote-python", default="python")
    parser.add_argument("--lingbot-map-root", default="", help="Optional remote lingbot-map checkout.")
    parser.add_argument("--local-output-dir", default="")
    parser.add_argument("--manifest-name", default="lingbot_job_manifest.json")
    parser.add_argument("--output-subdir", default="lingbot_output")
    parser.add_argument("--force-cpu", action="store_true", default=False)
    parser.add_argument("--no-offload-to-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(printable)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _remote_quote(path_or_arg: str) -> str:
    return shlex.quote(path_or_arg)


def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.exists():
        raise FileNotFoundError(f"Missing job directory: {job_dir}")
    manifest_path = job_dir / args.manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing job manifest: {manifest_path}")

    remote_root = args.remote_root.rstrip("/")
    remote_job_dir = f"{remote_root}/{job_dir.name}"
    remote_output_dir = f"{remote_job_dir}/{args.output_subdir}"
    local_output_dir = (
        Path(args.local_output_dir).expanduser().resolve()
        if args.local_output_dir
        else job_dir / args.output_subdir
    )

    mkdir_cmd = f"mkdir -p {_remote_quote(remote_root)}"
    _run(["ssh", args.remote, mkdir_cmd], dry_run=args.dry_run)
    _run(
        [
            "rsync",
            "-az",
            "--delete",
            f"{job_dir}/",
            f"{args.remote}:{remote_job_dir}/",
        ],
        dry_run=args.dry_run,
    )

    env_parts = [
        f"PYTHONPATH={_remote_quote(str(Path(args.remote_repo_root) / 'HMR3D/nuc/src'))}:$PYTHONPATH",
    ]
    if args.lingbot_map_root:
        env_parts.append(f"LINGBOT_MAP_ROOT={_remote_quote(args.lingbot_map_root)}")
    runner_args = [
        args.remote_python,
        str(Path(args.remote_repo_root) / "HMR3D/nuc/scripts/run_lingbot_dense_from_manifest.py"),
        "--manifest-json",
        f"{remote_job_dir}/{args.manifest_name}",
        "--model-path",
        args.model_path,
        "--output-dir",
        remote_output_dir,
    ]
    if args.force_cpu:
        runner_args.append("--force-cpu")
    if args.no_offload_to_cpu:
        runner_args.append("--no-offload-to-cpu")

    remote_cmd = (
        f"cd {_remote_quote(args.remote_repo_root)} && "
        f"{' '.join(env_parts)} "
        + " ".join(_remote_quote(part) for part in runner_args)
    )
    _run(["ssh", args.remote, remote_cmd], dry_run=args.dry_run)

    local_output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            f"{args.remote}:{remote_output_dir}/",
            f"{local_output_dir}/",
        ],
        dry_run=args.dry_run,
    )

    result = {
        "remote": args.remote,
        "local_job_dir": str(job_dir),
        "remote_job_dir": remote_job_dir,
        "remote_output_dir": remote_output_dir,
        "local_output_dir": str(local_output_dir),
        "predictions_npz": str(local_output_dir / "lingbot_predictions.npz"),
        "summary_json": str(local_output_dir / "lingbot_summary.json"),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
