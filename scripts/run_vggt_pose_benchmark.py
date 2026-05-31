import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "external_methods" / "vggt"))

from materialize_sparse_scene import ensure_colmap_text_model, load_split
from src.geometry.colmap import (
    align_similarity_umeyama,
    camera_centers_from_extrinsics,
    read_colmap_camera_centers,
    rmse,
    trajectory_extent,
)
from src.utils.io import ensure_dir, save_json


DEFAULT_SCENES = ["tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"]
DEFAULT_SPLITS = [
    "5views_uniform",
    "5views_random",
    "5views_quality",
    "5views_quality_diverse",
]

CSV_FIELDS = [
    "scene",
    "method",
    "split",
    "views",
    "selection",
    "status",
    "num_images",
    "device",
    "dtype",
    "load_images_sec",
    "infer_sec",
    "peak_gpu_memory_mb",
    "camera_center_rmse",
    "camera_center_mae",
    "camera_center_max_error",
    "reference_trajectory_extent",
    "normalized_camera_center_rmse",
    "alignment_scale",
    "depth_mean",
    "depth_median",
    "depth_conf_mean",
    "world_points_conf_mean",
    "output_dir",
    "error",
    "model_load_sec",
]


def append_csv(row: dict, path: Path) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def split_metadata(split_name: str) -> tuple[int | None, str]:
    prefix, _, selection = split_name.partition("views_")
    try:
        views = int(prefix)
    except ValueError:
        views = None
    return views, selection or split_name


def tensor_stats(tensor) -> dict:
    import torch

    if tensor is None:
        return {}
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    if not finite.any():
        return {"mean": None, "median": None, "min": None, "max": None}
    valid = value[finite]
    return {
        "mean": float(valid.mean().item()),
        "median": float(valid.median().item()),
        "min": float(valid.min().item()),
        "max": float(valid.max().item()),
    }


def image_paths_for_split(scene: str, split: str) -> list[Path]:
    names = load_split(scene, split)
    image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
    paths = [image_dir / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Split references missing images: {missing}")
    return paths


def reference_centers_for_split(scene: str, image_names: list[str]) -> np.ndarray:
    text_model = ensure_colmap_text_model(scene)
    centers = read_colmap_camera_centers(text_model / "images.txt")
    missing = [name for name in image_names if name not in centers]
    if missing:
        raise KeyError(f"Reference COLMAP centers missing for {scene}: {missing}")
    return np.stack([centers[name] for name in image_names], axis=0)


def run_one(model, scene: str, split: str, args) -> dict:
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = image_paths_for_split(scene, split)
    image_names = [path.name for path in paths]
    views, selection = split_metadata(split)
    output_dir = ensure_dir(PROJECT_ROOT / args.output_root / scene / "vggt_pose" / split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        dtype = torch.float16

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
    load_images_sec = time.perf_counter() - start

    infer_start = time.perf_counter()
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    infer_sec = time.perf_counter() - infer_start

    pose_enc = predictions["pose_enc"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    pred_centers = camera_centers_from_extrinsics(extrinsics.squeeze(0).detach().float().cpu().numpy())
    ref_centers = reference_centers_for_split(scene, image_names)
    aligned_centers, transform = align_similarity_umeyama(pred_centers, ref_centers)
    errors = np.linalg.norm(aligned_centers - ref_centers, axis=1)
    extent = trajectory_extent(ref_centers)

    np.save(output_dir / "vggt_camera_centers.npy", pred_centers)
    np.save(output_dir / "reference_camera_centers.npy", ref_centers)
    np.save(output_dir / "aligned_camera_centers.npy", aligned_centers)
    np.save(output_dir / "vggt_intrinsics.npy", intrinsics.squeeze(0).detach().float().cpu().numpy())

    depth_stats = tensor_stats(predictions.get("depth"))
    depth_conf_stats = tensor_stats(predictions.get("depth_conf"))
    world_conf_stats = tensor_stats(predictions.get("world_points_conf"))

    row = {
        "scene": scene,
        "method": "VGGT",
        "split": split,
        "views": views,
        "selection": selection,
        "status": "success",
        "num_images": len(paths),
        "device": device,
        "dtype": str(dtype),
        "load_images_sec": load_images_sec,
        "infer_sec": infer_sec,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else None
        ),
        "camera_center_rmse": rmse(errors),
        "camera_center_mae": float(np.mean(errors)),
        "camera_center_max_error": float(np.max(errors)),
        "reference_trajectory_extent": extent,
        "normalized_camera_center_rmse": None if extent < 1e-12 else rmse(errors) / extent,
        "alignment_scale": transform["scale"],
        "depth_mean": depth_stats.get("mean"),
        "depth_median": depth_stats.get("median"),
        "depth_conf_mean": depth_conf_stats.get("mean"),
        "world_points_conf_mean": world_conf_stats.get("mean"),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "error": None,
    }
    save_json(
        {
            "row": row,
            "image_names": image_names,
            "alignment": transform,
            "per_image_camera_center_error": dict(zip(image_names, errors.tolist())),
        },
        output_dir / "run_report.json",
    )
    return row


def failed_row(scene: str, split: str, error: Exception) -> dict:
    views, selection = split_metadata(split)
    return {
        "scene": scene,
        "method": "VGGT",
        "split": split,
        "views": views,
        "selection": selection,
        "status": "failed",
        "num_images": None,
        "device": None,
        "dtype": None,
        "load_images_sec": None,
        "infer_sec": None,
        "peak_gpu_memory_mb": None,
        "camera_center_rmse": None,
        "camera_center_mae": None,
        "camera_center_max_error": None,
        "reference_trajectory_extent": None,
        "normalized_camera_center_rmse": None,
        "alignment_scale": None,
        "depth_mean": None,
        "depth_median": None,
        "depth_conf_mean": None,
        "world_points_conf_mean": None,
        "output_dir": None,
        "error": repr(error),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--model-name", default="facebook/VGGT-1B")
    parser.add_argument("--output-root", default="results/method_outputs")
    parser.add_argument("--csv", default="results/csv/foundation/vggt_pose_benchmark.csv")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cache_dir = ensure_dir(PROJECT_ROOT / "external_methods" / "hf_cache")
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    import torch
    from vggt.models.vggt import VGGT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_start = time.perf_counter()
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()
    model_load_sec = time.perf_counter() - load_start

    csv_path = PROJECT_ROOT / args.csv
    rows = []
    for scene in args.scenes:
        for split in args.splits:
            report_path = PROJECT_ROOT / args.output_root / scene / "vggt_pose" / split / "run_report.json"
            if args.skip_existing and report_path.exists():
                data = json.loads(report_path.read_text(encoding="utf-8-sig"))
                row = data["row"]
                row["model_load_sec"] = model_load_sec
            else:
                try:
                    row = run_one(model, scene, split, args)
                except Exception as exc:
                    row = failed_row(scene, split, exc)
                row["model_load_sec"] = model_load_sec
                append_csv(row, csv_path)
            rows.append(row)
            print(json.dumps(row, indent=2))

    save_json({"model_name": args.model_name, "model_load_sec": model_load_sec, "rows": rows}, csv_path.with_suffix(".json"))
    print(f"Saved VGGT pose benchmark to {csv_path}")


if __name__ == "__main__":
    main()
