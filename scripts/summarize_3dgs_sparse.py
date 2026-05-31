import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


SPLIT_RE = re.compile(r"(?P<views>\d+)views_(?P<selection>.+)")


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["split_name"].str.extract(SPLIT_RE)
    df = df.copy()
    df["views"] = parsed["views"].astype(int)
    df["selection"] = parsed["selection"]
    return df.sort_values(["selection", "views"])


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for selection, group in df.groupby("selection"):
        group = group.sort_values("views")
        plt.plot(group["views"], group[metric], marker="o", linewidth=2, label=selection)
    plt.xlabel("Input views")
    plt.ylabel(ylabel)
    plt.legend(title="Selection")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/csv/3dgs_sparse_runs.csv")
    parser.add_argument("--output", default="results/csv/3dgs_sparse_summary.csv")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()

    df = pd.read_csv(PROJECT_ROOT / args.input)
    df = enrich(df)
    df = df.drop_duplicates(
        subset=["scene", "split_name", "iterations", "resolution"], keep="last"
    )
    output = PROJECT_ROOT / args.output
    ensure_dir(output.parent)
    df.to_csv(output, index=False)

    fig_dir = PROJECT_ROOT / args.fig_dir
    plot_metric(df, "train_psnr", "Train PSNR at final iteration", fig_dir / "3dgs_sparse_train_psnr.png")
    plot_metric(df, "runtime_sec", "Runtime (seconds)", fig_dir / "3dgs_sparse_runtime.png")
    plot_metric(df, "initial_points", "Filtered COLMAP points", fig_dir / "3dgs_sparse_initial_points.png")

    print(f"Saved summary CSV to {output}")
    print(f"Saved plots to {fig_dir}")
    print(df[["split_name", "views", "selection", "train_psnr", "runtime_sec", "initial_points"]].to_string(index=False))


if __name__ == "__main__":
    main()
