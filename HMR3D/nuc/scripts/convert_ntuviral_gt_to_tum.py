from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NTU VIRAL ground_truth.csv into a TUM trajectory file."
    )
    parser.add_argument("--gt-csv", required=True, help="Path to NTU VIRAL ground_truth.csv")
    parser.add_argument("--output-tum", required=True, help="Output TUM trajectory path")
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="Keep original absolute timestamps. Default behavior writes timestamps relative to the first GT sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_csv = Path(args.gt_csv).expanduser().resolve()
    output_tum = Path(args.output_tum).expanduser().resolve()
    output_tum.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    first_stamp_ns: float | None = None
    with gt_csv.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            stamp_ns = float(row["field.header.stamp"])
            if first_stamp_ns is None:
                first_stamp_ns = stamp_ns
            if args.absolute_time:
                timestamp = stamp_ns * 1e-9
            else:
                timestamp = (stamp_ns - first_stamp_ns) * 1e-9
            tx = float(row["field.pose.position.x"])
            ty = float(row["field.pose.position.y"])
            tz = float(row["field.pose.position.z"])
            qx = float(row["field.pose.orientation.x"])
            qy = float(row["field.pose.orientation.y"])
            qz = float(row["field.pose.orientation.z"])
            qw = float(row["field.pose.orientation.w"])
            rows.append(
                f"{timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
            )

    output_tum.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"wrote {len(rows)} poses to {output_tum}")


if __name__ == "__main__":
    main()
