import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, save_csv, save_json


INSTANTSPLAT_ROOT = PROJECT_ROOT / "external_methods" / "InstantSplat"
DEFAULT_CKPT = (
    INSTANTSPLAT_ROOT
    / "mast3r"
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)


def load_split(scene: str, split: str) -> list[str]:
    split_path = PROJECT_ROOT / "datasets" / "processed" / scene / "splits" / f"{Path(scene).name}_splits.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    try:
        return data["splits"][split]
    except KeyError as exc:
        raise KeyError(f"Split {split!r} not found in {split_path}") from exc


def materialize_source(scene: str, split: str, image_names: list[str]) -> Path:
    raw_image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
    if not raw_image_dir.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_image_dir}")

    source_dir = PROJECT_ROOT / "datasets" / "instantsplat_smoke" / scene / split
    image_dir = ensure_dir(source_dir / "images")

    for image_name in image_names:
        src = raw_image_dir / image_name
        if not src.exists():
            raise FileNotFoundError(f"Selected image not found: {src}")
        shutil.copy2(src, image_dir / image_name)

    save_json(
        {
            "scene": scene,
            "split": split,
            "images": image_names,
            "source_dir": str(source_dir),
        },
        source_dir / "materialize_report.json",
    )
    return source_dir


def patch_torch_load_for_trusted_checkpoint():
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    return original_load


def run_init_geo(args, source_dir: Path, output_dir: Path) -> dict:
    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(f"InstantSplat MASt3R checkpoint not found: {DEFAULT_CKPT}")

    sys.path.insert(0, str(INSTANTSPLAT_ROOT))
    import init_geo

    # init_geo.main references a module-level args object in two places.
    init_geo.args = SimpleNamespace(device=args.device, focal_avg=False)

    original_load = patch_torch_load_for_trusted_checkpoint()
    start = time.perf_counter()
    try:
        init_geo.main(
            source_path=str(source_dir),
            model_path=str(output_dir),
            ckpt_path=str(DEFAULT_CKPT),
            device=args.device,
            batch_size=1,
            image_size=args.image_size,
            schedule="cosine",
            lr=0.01,
            niter=args.geo_niter,
            min_conf_thr=5,
            llffhold=8,
            n_views=args.views,
            co_vis_dsp=True,
            depth_thre=0.01,
            conf_aware_ranking=False,
            focal_avg=False,
            infer_video=True,
        )
    finally:
        torch.load = original_load
    elapsed = time.perf_counter() - start

    sparse_dir = source_dir / f"sparse_{args.views}" / "0"
    return {
        "init_geo_sec": elapsed,
        "sparse_dir": str(sparse_dir),
        "points3d_exists": (sparse_dir / "points3D.txt").exists()
        or (sparse_dir / "points3D.ply").exists(),
        "confidence_exists": (sparse_dir / "confidence_dsp.npy").exists(),
    }


def run_train(args, source_dir: Path, output_dir: Path) -> dict:
    if args.train_iterations <= 0:
        return {"train_status": "skipped", "train_sec": 0.0}

    log_path = output_dir / "instantsplat_train_console.log"
    command = [
        sys.executable,
        "train.py",
        "--source_path",
        str(source_dir),
        "--model_path",
        str(output_dir),
        "--n_views",
        str(args.views),
        "--iterations",
        str(args.train_iterations),
        "--resolution",
        str(args.resolution),
        "--test_iterations",
        str(args.train_iterations),
        "--save_iterations",
        str(args.train_iterations),
        "--disable_viewer",
    ]

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            command,
            cwd=INSTANTSPLAT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    point_cloud = output_dir / "point_cloud" / f"iteration_{args.train_iterations}" / "point_cloud.ply"
    return {
        "train_status": "success" if completed.returncode == 0 else "failed",
        "train_returncode": completed.returncode,
        "train_sec": elapsed,
        "train_log": str(log_path),
        "saved_point_cloud": str(point_cloud),
        "saved_point_cloud_exists": point_cloud.exists(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="tandt/train")
    parser.add_argument("--split", default="5views_quality_diverse")
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--geo-niter", type=int, default=300)
    parser.add_argument("--train-iterations", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--csv", default="results/csv/foundation/instantsplat_smoke.csv")
    args = parser.parse_args()

    output_dir = ensure_dir(PROJECT_ROOT / "results" / "method_outputs" / args.scene / "instantsplat_smoke" / args.split)
    image_names = load_split(args.scene, args.split)
    if len(image_names) != args.views:
        raise ValueError(f"Split {args.split} has {len(image_names)} images, not {args.views}")

    row = {
        "scene": args.scene,
        "method": "InstantSplat",
        "split": args.split,
        "views": args.views,
        "status": "started",
        "device": args.device,
        "image_size": args.image_size,
        "train_iterations": args.train_iterations,
        "output_dir": str(output_dir),
        "error": "",
    }

    try:
        source_dir = materialize_source(args.scene, args.split, image_names)
        row["source_dir"] = str(source_dir)
        row.update(run_init_geo(args, source_dir, output_dir))
        row.update(run_train(args, source_dir, output_dir))
        row["status"] = "success" if row.get("train_status") != "failed" else "partial"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        raise
    finally:
        save_json(row, output_dir / "run_report.json")
        save_csv([row], PROJECT_ROOT / args.csv)
        print(f"Saved InstantSplat smoke report to {output_dir / 'run_report.json'}")
        print(f"Saved InstantSplat smoke CSV to {PROJECT_ROOT / args.csv}")


if __name__ == "__main__":
    main()
