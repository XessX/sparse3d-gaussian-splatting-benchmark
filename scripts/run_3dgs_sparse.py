import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from materialize_sparse_scene import materialize_sparse_scene
from src.utils.io import ensure_dir, save_json


TRAIN_METRIC_RE = re.compile(
    r"\[ITER\s+(?P<iteration>\d+)\]\s+Evaluating train:\s+L1\s+"
    r"(?P<l1>[0-9.eE+-]+)\s+PSNR\s+(?P<psnr>[0-9.eE+-]+)"
)


def append_csv(row: dict, path: Path) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_train_metrics(output: str) -> dict:
    matches = list(TRAIN_METRIC_RE.finditer(output))
    if not matches:
        return {}
    last = matches[-1]
    return {
        "metric_iteration": int(last.group("iteration")),
        "train_l1": float(last.group("l1")),
        "train_psnr": float(last.group("psnr")),
    }


def run_one(scene: str, split_name: str, iterations: int, resolution: int, csv_path: Path) -> dict:
    materialized = materialize_sparse_scene(scene, split_name, "datasets/sparse_scenes")
    source_path = Path(materialized["output_dir"])
    model_path = PROJECT_ROOT / "results" / "method_outputs" / scene / "3dgs_sparse" / split_name
    ensure_dir(model_path)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "external_methods" / "gaussian-splatting" / "train.py"),
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--test_iterations",
        str(iterations),
        "--disable_viewer",
        "--resolution",
        str(resolution),
    ]

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    runtime_sec = time.perf_counter() - start
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = model_path / "train_console.log"
    log_path.write_text(output, encoding="utf-8")

    metrics = parse_train_metrics(output)
    point_cloud = model_path / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    report = {
        "scene": scene,
        "method": "3dgs_sparse",
        "split_name": split_name,
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "source_path": str(source_path.relative_to(PROJECT_ROOT)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "iterations": iterations,
        "resolution": resolution,
        "runtime_sec": runtime_sec,
        "num_images": materialized["num_images"],
        "initial_points": materialized["num_observed_points"],
        "metric_iteration": metrics.get("metric_iteration"),
        "train_l1": metrics.get("train_l1"),
        "train_psnr": metrics.get("train_psnr"),
        "saved_point_cloud": str(point_cloud.relative_to(PROJECT_ROOT)) if point_cloud.exists() else "",
        "point_cloud_bytes": point_cloud.stat().st_size if point_cloud.exists() else 0,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
    }
    save_json(report, model_path / "run_report.json")
    append_csv(report, csv_path)

    if proc.returncode != 0:
        raise RuntimeError(f"3DGS sparse run failed for {split_name}. See {log_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--csv", default="results/csv/3dgs_sparse_runs.csv")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    reports = [
        run_one(args.scene, split_name, args.iterations, args.resolution, csv_path)
        for split_name in args.splits
    ]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
