import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rendered_images import evaluate
from materialize_sparse_scene import load_split, materialize_sparse_scene
from run_3dgs_sparse import TRAIN_METRIC_RE, parse_train_metrics
from src.degradation.image_degradations import adjust_exposure, apply_motion_blur
from src.utils.io import ensure_dir, save_json


DEFAULT_SCENES = ["tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"]
DEFAULT_SPLITS = [
    "5views_uniform",
    "5views_random",
    "5views_quality",
    "5views_quality_diverse",
]
DEFAULT_CONDITIONS = ["jpeg_q20", "jpeg_q40", "blur_k15", "lowres_50", "under_0.6", "under_0.4", "over_1.4"]


def append_csv(row: dict, path: Path) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def degrade_image(input_path: Path, output_path: Path, condition: str) -> None:
    image = Image.open(input_path).convert("RGB")
    rgb = np.asarray(image)
    ensure_dir(output_path.parent)

    if condition.startswith("jpeg_q"):
        quality = int(condition.replace("jpeg_q", ""))
        image.save(output_path, quality=quality)
        return
    if condition == "blur_k15":
        Image.fromarray(apply_motion_blur(rgb, 15)).save(output_path)
        return
    if condition.startswith("under_"):
        alpha = float(condition.replace("under_", ""))
        Image.fromarray(adjust_exposure(rgb, alpha)).save(output_path)
        return
    if condition.startswith("over_"):
        alpha = float(condition.replace("over_", ""))
        Image.fromarray(adjust_exposure(rgb, alpha)).save(output_path)
        return
    if condition == "lowres_50":
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        Image.fromarray(restored).save(output_path)
        return
    raise ValueError(f"Unknown degradation condition: {condition}")


def apply_train_degradation(scene: str, split: str, scene_dir: Path, condition: str) -> None:
    train_names = load_split(scene, split)
    raw_image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
    output_image_dir = scene_dir / "images"
    for name in train_names:
        degrade_image(raw_image_dir / name, output_image_dir / name, condition)


def prepare_scene(
    scene: str,
    split: str,
    condition: str,
    output_root: str,
    include_test: bool,
) -> dict:
    test_path = f"datasets/processed/{scene}/eval_test_views.json" if include_test else None
    report = materialize_sparse_scene(scene, split, output_root, test_path)
    apply_train_degradation(scene, split, Path(report["output_dir"]), condition)
    report["condition"] = condition
    save_json(report, Path(report["output_dir"]) / "degradation_materialize_report.json")
    return report


def run_train(scene: str, split: str, condition: str, iterations: int, resolution: int) -> dict:
    materialized = prepare_scene(
        scene,
        split,
        condition,
        f"datasets/degraded_scenes/{condition}",
        include_test=False,
    )
    source_path = Path(materialized["output_dir"])
    model_path = PROJECT_ROOT / "results" / "method_outputs" / scene / "3dgs_degraded" / condition / split
    if model_path.exists():
        shutil.rmtree(model_path)
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
        "method": "3dgs_degraded",
        "condition": condition,
        "split_name": split,
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
    if proc.returncode != 0:
        raise RuntimeError(f"3DGS degraded run failed for {scene} {condition} {split}. See {log_path}")
    return report


def render_eval(scene: str, split: str, condition: str, iterations: int, resolution: int) -> dict:
    materialized = prepare_scene(
        scene,
        split,
        condition,
        f"datasets/degraded_eval_scenes/{condition}",
        include_test=True,
    )
    source_path = Path(materialized["output_dir"])
    model_path = PROJECT_ROOT / "results" / "method_outputs" / scene / "3dgs_degraded" / condition / split
    command = [
        sys.executable,
        str(PROJECT_ROOT / "external_methods" / "gaussian-splatting" / "render.py"),
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--iteration",
        str(iterations),
        "--skip_train",
        "--eval",
        "--resolution",
        str(resolution),
    ]
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path = model_path / "render_console.log"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Render failed for {scene} {condition} {split}. See {log_path}")

    summary = evaluate(model_path, f"ours_{iterations}")
    report = {
        "scene": scene,
        "condition": condition,
        "split": split,
        "iteration": iterations,
        "resolution": resolution,
        "render_returncode": proc.returncode,
        "eval": summary,
    }
    save_json(report, model_path / "render_eval_report.json")
    return report


def run_one(scene: str, split: str, condition: str, iterations: int, resolution: int, csv_path: Path) -> dict:
    train_report = run_train(scene, split, condition, iterations, resolution)
    eval_report = render_eval(scene, split, condition, iterations, resolution)
    row = {
        **train_report,
        "num_test_images": eval_report["eval"].get("num_images"),
        "eval_psnr": eval_report["eval"].get("psnr"),
        "eval_ssim": eval_report["eval"].get("ssim"),
        "eval_lpips": eval_report["eval"].get("lpips"),
        "metrics_summary": eval_report["eval"].get("metrics_summary", ""),
        "per_view_csv": eval_report["eval"].get("per_view_csv", ""),
    }
    append_csv(row, csv_path)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--csv", default="results/csv/3dgs_degradation_runs_3000.csv")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    rows = []
    for scene in args.scenes:
        for condition in args.conditions:
            for split in args.splits:
                rows.append(run_one(scene, split, condition, args.iterations, args.resolution, csv_path))
                print(json.dumps(rows[-1], indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
