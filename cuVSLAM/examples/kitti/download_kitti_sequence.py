import argparse
import shutil
from pathlib import Path

from remotezip import RemoteZip


ODOMETRY_GRAY_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a single KITTI odometry grayscale sequence without fetching the full 22GB archive."
    )
    parser.add_argument(
        "--sequence",
        default="06",
        help="KITTI odometry sequence id, e.g. 06.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent),
        help="Root folder where dataset/sequences/<sequence> will be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing extracted sequence before downloading again.",
    )
    return parser.parse_args()


def main():
    """Download the selected KITTI sequence from the public mirror zip."""
    args = parse_args()
    sequence = f"{int(args.sequence):02d}"
    output_root = Path(args.output_root).expanduser().resolve()
    target_root = output_root / "dataset" / "sequences" / sequence
    prefix = f"dataset/sequences/{sequence}/"

    if args.force and target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    with RemoteZip(ODOMETRY_GRAY_URL) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith(prefix) and not name.endswith("/")
        ]
        print(
            f"Downloading KITTI sequence {sequence}: {len(members)} files to {target_root}",
            flush=True,
        )
        for idx, member in enumerate(members, start=1):
            relative = Path(member).relative_to(prefix)
            destination = target_root / relative
            temp_destination = destination.with_suffix(destination.suffix + ".part")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(temp_destination, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            temp_destination.replace(destination)
            if idx == 1 or idx % 100 == 0 or idx == len(members):
                print(f"{idx}/{len(members)}: {relative}", flush=True)


if __name__ == "__main__":
    main()
