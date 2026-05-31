import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "external_methods" / "mast3r"))
sys.path.insert(0, str(PROJECT_ROOT / "external_methods" / "mast3r" / "dust3r"))

from materialize_sparse_scene import ensure_colmap_text_model, load_split
from src.geometry.colmap import align_similarity_umeyama, read_colmap_camera_centers, rmse, trajectory_extent
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
    "image_size",
    "niter1",
    "niter2",
    "device",
    "model_load_sec",
    "image_load_sec",
    "pair_count",
    "alignment_sec",
    "peak_gpu_memory_mb",
    "camera_center_rmse",
    "camera_center_mae",
    "camera_center_max_error",
    "reference_trajectory_extent",
    "normalized_camera_center_rmse",
    "alignment_scale",
    "focal_mean",
    "output_dir",
    "error",
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


def failed_row(scene: str, split: str, args, error: Exception, model_load_sec: float | None = None) -> dict:
    views, selection = split_metadata(split)
    return {
        "scene": scene,
        "method": "MASt3R",
        "split": split,
        "views": views,
        "selection": selection,
        "status": "failed",
        "image_size": args.image_size,
        "niter1": args.niter1,
        "niter2": args.niter2,
        "device": args.device,
        "model_load_sec": model_load_sec,
        "error": repr(error),
    }


def run_one(model, scene: str, split: str, args, model_load_sec: float) -> dict:
    import torch
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs

    paths = image_paths_for_split(scene, split)
    filelist = [str(path) for path in paths]
    image_names = [path.name for path in paths]
    views, selection = split_metadata(split)
    output_dir = ensure_dir(PROJECT_ROOT / args.output_root / scene / "mast3r_pose" / split)
    cache_dir = output_dir / "cache"
    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    ensure_dir(cache_dir)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    image_start = time.perf_counter()
    images = load_images(filelist, size=args.image_size, verbose=False)
    image_load_sec = time.perf_counter() - image_start
    pairs = make_pairs(images, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)

    align_start = time.perf_counter()
    sparse_ga = sparse_global_alignment(
        filelist,
        pairs,
        str(cache_dir),
        model,
        subsample=args.subsample,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        device=args.device,
        opt_depth=args.opt_depth,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
    )
    alignment_sec = time.perf_counter() - align_start

    poses = sparse_ga.get_im_poses().detach().float().cpu().numpy()
    pred_centers = poses[:, :3, 3]
    ref_centers = reference_centers_for_split(scene, image_names)
    aligned_centers, transform = align_similarity_umeyama(pred_centers, ref_centers)
    errors = np.linalg.norm(aligned_centers - ref_centers, axis=1)
    extent = trajectory_extent(ref_centers)
    focals = sparse_ga.get_focals().detach().float().cpu().numpy()

    np.save(output_dir / "mast3r_cam_to_world.npy", poses)
    np.save(output_dir / "mast3r_camera_centers.npy", pred_centers)
    np.save(output_dir / "reference_camera_centers.npy", ref_centers)
    np.save(output_dir / "aligned_camera_centers.npy", aligned_centers)
    np.save(output_dir / "mast3r_focals.npy", focals)

    row = {
        "scene": scene,
        "method": "MASt3R",
        "split": split,
        "views": views,
        "selection": selection,
        "status": "success",
        "num_images": len(paths),
        "image_size": args.image_size,
        "niter1": args.niter1,
        "niter2": args.niter2,
        "device": args.device,
        "model_load_sec": model_load_sec,
        "image_load_sec": image_load_sec,
        "pair_count": len(pairs),
        "alignment_sec": alignment_sec,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device == "cuda" else None
        ),
        "camera_center_rmse": rmse(errors),
        "camera_center_mae": float(np.mean(errors)),
        "camera_center_max_error": float(np.max(errors)),
        "reference_trajectory_extent": extent,
        "normalized_camera_center_rmse": None if extent < 1e-12 else rmse(errors) / extent,
        "alignment_scale": transform["scale"],
        "focal_mean": float(np.mean(focals)),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument(
        "--weights",
        default="external_methods/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--scene-graph", default="complete")
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--niter1", type=int, default=80)
    parser.add_argument("--niter2", type=int, default=40)
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--matching-conf-thr", type=float, default=5.0)
    parser.add_argument("--opt-depth", action="store_true")
    parser.add_argument("--shared-intrinsics", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="results/method_outputs")
    parser.add_argument("--csv", default="results/csv/foundation/mast3r_pose_5views.csv")
    args = parser.parse_args()

    import torch
    from mast3r.model import AsymmetricMASt3R

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = PROJECT_ROOT / args.weights
    if not weights.exists():
        raise FileNotFoundError(f"MASt3R checkpoint not found: {weights}")

    original_torch_load = torch.load

    def torch_load_trusted_checkpoint(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_trusted_checkpoint
    load_start = time.perf_counter()
    try:
        model = AsymmetricMASt3R.from_pretrained(str(weights)).to(args.device)
    finally:
        torch.load = original_torch_load
    model.eval()
    model_load_sec = time.perf_counter() - load_start

    csv_path = PROJECT_ROOT / args.csv
    rows = []
    for scene in args.scenes:
        for split in args.splits:
            try:
                row = run_one(model, scene, split, args, model_load_sec)
            except Exception as exc:
                row = failed_row(scene, split, args, exc, model_load_sec)
            append_csv(row, csv_path)
            rows.append(row)
            print(json.dumps(row, indent=2))

    save_json({"weights": str(weights), "rows": rows}, csv_path.with_suffix(".json"))
    print(f"Saved MASt3R pose benchmark to {csv_path}")


if __name__ == "__main__":
    main()
