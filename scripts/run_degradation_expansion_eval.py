import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_3dgs_degradation import run_one as run_degradation_one


DEFAULT_SCENES = ["tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"]
DEFAULT_SPLITS = ["5views_uniform", "5views_random", "5views_quality", "5views_quality_diverse"]
DEFAULT_CONDITIONS = ["jpeg_q40", "under_0.4", "over_1.4"]


def metrics_path(scene: str, condition: str, split: str, iteration: int) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "method_outputs"
        / scene
        / "3dgs_degraded"
        / condition
        / split
        / "test"
        / f"ours_{iteration}"
        / "metrics_summary.json"
    )


def refresh_degradation_summary() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "collect_degradation_results.py")],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "create_degradation_expansion_todo.py")],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--csv", default="results/csv/3dgs_degradation_runs_3000.csv")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    pending = []
    for scene in args.scenes:
        for condition in args.conditions:
            for split in args.splits:
                if not metrics_path(scene, condition, split, args.iterations).exists():
                    pending.append((scene, condition, split))
    if args.max_runs is not None:
        pending = pending[: args.max_runs]

    print(f"Pending expanded degradation evaluations to run: {len(pending)}")
    csv_path = PROJECT_ROOT / args.csv
    for index, (scene, condition, split) in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {scene} / {condition} / {split}")
        row = run_degradation_one(scene, split, condition, args.iterations, args.resolution, csv_path)
        print(
            f"  PSNR={row.get('eval_psnr'):.3f} "
            f"SSIM={row.get('eval_ssim'):.3f} "
            f"LPIPS={row.get('eval_lpips'):.3f}"
        )
        refresh_degradation_summary()

    refresh_degradation_summary()
    print("Expanded degradation evaluation complete for selected rows.")


if __name__ == "__main__":
    main()
