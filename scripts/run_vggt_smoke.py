import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import list_images
from src.utils.io import ensure_dir


def tensor_summary(value):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {k: tensor_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_summary(v) for v in value[:8]]
    return {"type": type(value).__name__}


def save_normalized_map(array: np.ndarray, path: Path) -> None:
    finite = np.isfinite(array)
    if not finite.any():
        Image.fromarray(np.zeros(array.shape, dtype=np.uint8)).save(path)
        return
    valid = array[finite]
    lo, hi = np.percentile(valid, [2, 98])
    if hi - lo < 1e-9:
        norm = np.zeros(array.shape, dtype=np.uint8)
    else:
        norm = np.clip((array - lo) / (hi - lo), 0, 1)
        norm = (norm * 255).astype(np.uint8)
    Image.fromarray(norm).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="demo_scene")
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--model-name", default="facebook/VGGT-1B")
    parser.add_argument("--output-root", default="results/method_outputs")
    args = parser.parse_args()

    cache_dir = ensure_dir(PROJECT_ROOT / "external_methods" / "hf_cache")
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    image_dir = PROJECT_ROOT / "datasets" / "raw" / args.scene / "images"
    image_paths = [str(p) for p in list_images(image_dir)[: args.num_images]]
    output_dir = ensure_dir(PROJECT_ROOT / args.output_root / args.scene / "vggt_smoke")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        dtype = torch.float16

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()
    load_sec = time.perf_counter() - start

    images = load_and_preprocess_images(image_paths).to(device)
    infer_start = time.perf_counter()
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    infer_sec = time.perf_counter() - infer_start

    report = {
        "scene": args.scene,
        "method": "vggt_smoke",
        "model_name": args.model_name,
        "image_paths": image_paths,
        "device": device,
        "dtype": str(dtype),
        "load_sec": load_sec,
        "infer_sec": infer_sec,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device == "cuda"
            else None
        ),
        "input_summary": tensor_summary(images),
        "prediction_summary": tensor_summary(predictions),
    }

    preview_files = []
    if "depth" in predictions:
        depth = predictions["depth"].detach().float().cpu().numpy()[0, :, :, :, 0]
        for idx in range(depth.shape[0]):
            path = output_dir / f"depth_view_{idx:03d}.png"
            save_normalized_map(depth[idx], path)
            preview_files.append(str(path))
    if "depth_conf" in predictions:
        conf = predictions["depth_conf"].detach().float().cpu().numpy()[0]
        for idx in range(conf.shape[0]):
            path = output_dir / f"depth_conf_view_{idx:03d}.png"
            save_normalized_map(conf[idx], path)
            preview_files.append(str(path))

    report["preview_files"] = preview_files

    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
