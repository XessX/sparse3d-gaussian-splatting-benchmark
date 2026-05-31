import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.manifest import build_manifest
from src.utils.io import ensure_dir, save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--methods", nargs="+", default=["placeholder_3dgs"])
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 5, 8, 12, 20])
    parser.add_argument("--selections", nargs="+", default=["uniform", "random", "quality"])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clean", "resize_50", "jpeg_q40", "motion_blur_k7", "exposure_under_0.6"],
    )
    parser.add_argument("--output-csv", default="results/csv/experiment_manifest.csv")
    parser.add_argument("--output-json", default="results/logs/experiment_manifest.json")
    args = parser.parse_args()

    rows = build_manifest(
        scene=args.scene,
        methods=args.methods,
        view_counts=args.views,
        selections=args.selections,
        conditions=args.conditions,
    )

    save_csv(rows, args.output_csv)
    output_json = Path(args.output_json)
    ensure_dir(output_json.parent)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    missing = sum(row["status"] == "missing_input" for row in rows)
    print(f"Saved {len(rows)} experiments to {args.output_csv}")
    print(f"Saved JSON manifest to {args.output_json}")
    print(f"Rows with missing inputs: {missing}")


if __name__ == "__main__":
    main()
