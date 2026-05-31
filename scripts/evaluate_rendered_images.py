import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

from src.utils.io import save_json


def image_to_float(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_lpips_rows(rows: list[dict], render_paths: list[Path], gt_paths: list[Path]) -> None:
    try:
        import torch
        import lpips
    except Exception as exc:
        for row in rows:
            row["lpips"] = None
            row["lpips_error"] = repr(exc)
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = lpips.LPIPS(net="vgg").to(device)
        loss_fn.eval()
    except Exception as exc:
        for row in rows:
            row["lpips"] = None
            row["lpips_error"] = repr(exc)
        return

    for row, render_path, gt_path in zip(rows, render_paths, gt_paths):
        pred = image_to_float(render_path)
        gt = image_to_float(gt_path)
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            row["lpips"] = float(loss_fn(pred_t, gt_t).item())
        row["lpips_error"] = None


def evaluate(model_path: Path, method: str | None = None) -> dict:
    test_dir = model_path / "test"
    if method is None:
        methods = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
        if not methods:
            raise FileNotFoundError(f"No rendered method folders found in {test_dir}")
        method = methods[-1]

    method_dir = test_dir / method
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not render_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing renders/gt directories under {method_dir}")

    render_paths = sorted(render_dir.glob("*.png"))
    gt_paths = [gt_dir / p.name for p in render_paths]
    if not render_paths:
        raise FileNotFoundError(f"No rendered PNGs found in {render_dir}")

    rows = []
    for render_path, gt_path in zip(render_paths, gt_paths):
        pred = image_to_float(render_path)
        gt = image_to_float(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {render_path.name}: {pred.shape} vs {gt.shape}")
        rows.append(
            {
                "image": render_path.name,
                "psnr": float(peak_signal_noise_ratio(gt, pred, data_range=1.0)),
                "ssim": float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0)),
            }
        )

    compute_lpips_rows(rows, render_paths, gt_paths)
    df = pd.DataFrame(rows)
    summary = {
        "model_path": str(model_path),
        "method": method,
        "num_images": len(rows),
        "psnr": float(df["psnr"].mean()),
        "ssim": float(df["ssim"].mean()),
        "lpips": None if df["lpips"].isna().all() else float(df["lpips"].dropna().mean()),
        "per_view_csv": str(method_dir / "metrics_per_view.csv"),
    }
    df.to_csv(method_dir / "metrics_per_view.csv", index=False)
    save_json(summary, method_dir / "metrics_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", default=None)
    args = parser.parse_args()

    summary = evaluate(PROJECT_ROOT / args.model_path, args.method)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
