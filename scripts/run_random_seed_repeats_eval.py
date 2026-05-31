import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from render_eval_3dgs import render_split
from run_3dgs_sparse import run_one as run_sparse_one


def metrics_path(scene: str, split: str, iteration: int) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "method_outputs"
        / scene
        / "3dgs_sparse"
        / split
        / "test"
        / f"ours_{iteration}"
        / "metrics_summary.json"
    )


def point_cloud_path(scene: str, split: str, iteration: int) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "method_outputs"
        / scene
        / "3dgs_sparse"
        / split
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )


def refresh_random_summary() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "create_random_seed_repeats.py")],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/csv/random_seed_repeats.csv")
    parser.add_argument("--train-csv", default="results/csv/3dgs_random_seed_repeat_runs_3000.csv")
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--views", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    refresh_random_summary()
    df = pd.read_csv(PROJECT_ROOT / args.csv)
    if args.scenes:
        df = df[df["scene"].isin(args.scenes)]
    if args.views:
        df = df[df["views"].isin(args.views)]
    if args.seeds:
        df = df[df["seed"].isin(args.seeds)]

    pending = []
    for _, row in df.sort_values(["scene", "seed", "views"]).iterrows():
        scene = str(row["scene"])
        split = str(row["split"])
        if metrics_path(scene, split, args.iteration).exists():
            continue
        pending.append(row)

    if args.max_runs is not None:
        pending = pending[: args.max_runs]

    print(f"Pending random-seed evaluations to run: {len(pending)}")
    train_csv = PROJECT_ROOT / args.train_csv
    for index, row in enumerate(pending, start=1):
        scene = str(row["scene"])
        split = str(row["split"])
        print(f"[{index}/{len(pending)}] {scene} / {split}")
        if not point_cloud_path(scene, split, args.iteration).exists():
            run_sparse_one(scene, split, args.iteration, args.resolution, train_csv)
        render_split(scene, split, args.iteration, args.resolution, skip_existing=True)
        summary = json.loads(metrics_path(scene, split, args.iteration).read_text(encoding="utf-8"))
        print(
            f"  PSNR={summary.get('psnr'):.3f} "
            f"SSIM={summary.get('ssim'):.3f} "
            f"LPIPS={summary.get('lpips'):.3f}"
        )
        refresh_random_summary()

    refresh_random_summary()
    print("Random-seed repeat evaluation complete for selected rows.")


if __name__ == "__main__":
    main()
