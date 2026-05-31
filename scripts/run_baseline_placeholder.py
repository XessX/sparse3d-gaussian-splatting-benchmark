import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_csv, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", default="placeholder_3dgs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Placeholder output so the plotting/reporting pipeline can be tested
    # before installing heavy external methods.
    rows = []
    for views in [2, 3, 5, 8, 12, 20]:
        rows.append({
            "scene": args.scene,
            "method": args.method,
            "views": views,
            "condition": "clean",
            "psnr": 18 + views * 0.45 + rng.uniform(-0.4, 0.4),
            "ssim": min(0.95, 0.55 + views * 0.015 + rng.uniform(-0.02, 0.02)),
            "lpips": max(0.05, 0.42 - views * 0.012 + rng.uniform(-0.02, 0.02)),
            "runtime_sec": 100 + views * 4,
            "gpu_memory_mb": 6000,
            "failure": 0
        })

    out_path = Path("results/csv") / f"{args.scene}_{args.method}_placeholder_results.csv"
    ensure_dir(out_path.parent)
    save_csv(rows, out_path)
    print(f"Saved placeholder results to {out_path}")

if __name__ == "__main__":
    main()
