import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


SPLIT_RE = re.compile(r"(?P<views>\d+)views_(?P<selection>.+)")


def parse_split(split: str) -> tuple[int | None, str]:
    match = SPLIT_RE.fullmatch(split)
    if not match:
        return None, split
    return int(match.group("views")), match.group("selection")


def scene_from_model_path(model_path: Path) -> str:
    marker = ("results", "method_outputs")
    parts = model_path.parts
    for idx in range(len(parts) - len(marker)):
        if parts[idx : idx + len(marker)] == marker:
            scene_parts = parts[idx + len(marker) : -2]
            return "/".join(scene_parts)
    try:
        rel = model_path.relative_to(PROJECT_ROOT / "results" / "method_outputs")
        return "/".join(rel.parts[:-2])
    except ValueError:
        return ""


def collect_train_rows(root: Path, iteration: int) -> pd.DataFrame:
    rows = []
    for report_path in sorted(root.glob("**/3dgs_sparse/*/run_report.json")):
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        if int(report.get("iterations", -1)) != iteration:
            continue
        split = report.get("split_name") or report_path.parent.name
        views, selection = parse_split(split)
        row = {
            "scene": report.get("scene") or scene_from_model_path(report_path.parent),
            "split": split,
            "views": views,
            "selection": selection,
            "iterations": report.get("iterations"),
            "resolution": report.get("resolution"),
            "status": report.get("status"),
            "runtime_sec": report.get("runtime_sec"),
            "train_psnr": report.get("train_psnr"),
            "train_l1": report.get("train_l1"),
            "initial_points": report.get("initial_points"),
            "model_path": report.get("model_path"),
            "run_report": str(report_path),
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No run_report.json files found under {root}")
    return pd.DataFrame(rows).sort_values(["scene", "views", "selection"])


def collect_eval_rows(root: Path, iteration: int) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(root.glob(f"**/3dgs_sparse/*/test/ours_{iteration}/metrics_summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        model_path = Path(summary["model_path"])
        split = model_path.name
        views, selection = parse_split(split)
        rows.append(
            {
                "scene": scene_from_model_path(model_path),
                "split": split,
                "views": views,
                "selection": selection,
                "iteration": iteration,
                "num_test_images": summary.get("num_images"),
                "psnr": summary.get("psnr"),
                "ssim": summary.get("ssim"),
                "lpips": summary.get("lpips"),
                "metrics_summary": str(summary_path),
                "per_view_csv": summary.get("per_view_csv"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics_summary.json files found under {root}")
    return pd.DataFrame(rows).sort_values(["scene", "views", "selection"])


def plot_metric_mean(eval_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    grouped = (
        eval_df.dropna(subset=[metric])
        .groupby(["selection", "views"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"), count=(metric, "count"))
    )
    plt.figure(figsize=(7.5, 5))
    for selection, group in grouped.groupby("selection"):
        group = group.sort_values("views")
        std = group["std"].fillna(0.0)
        label = selection.replace("quality_diverse", "quality+diverse")
        plt.plot(group["views"], group["mean"], marker="o", linewidth=2, label=label)
        plt.fill_between(group["views"], group["mean"] - std, group["mean"] + std, alpha=0.15)
    plt.xlabel("Number of training views")
    plt.ylabel(ylabel)
    plt.title(f"Multi-scene {ylabel} across sparse-view settings")
    plt.legend(title="Selection")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def export_mean_table(eval_df: pd.DataFrame, output: Path) -> pd.DataFrame:
    summary = (
        eval_df.groupby(["views", "selection"], as_index=False)
        .agg(
            scenes=("scene", "nunique"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            lpips_mean=("lpips", "mean"),
            lpips_std=("lpips", "std"),
        )
        .sort_values(["views", "selection"])
    )
    ensure_dir(output.parent)
    summary.to_csv(output, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/method_outputs")
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--train-csv", default="results/csv/3dgs_multiscene_train_3000.csv")
    parser.add_argument("--eval-csv", default="results/csv/3dgs_multiscene_eval_3000.csv")
    parser.add_argument("--mean-csv", default="results/csv/3dgs_multiscene_eval_mean_3000.csv")
    parser.add_argument("--figures", default="results/figures/3dgs_multiscene")
    args = parser.parse_args()

    root = PROJECT_ROOT / args.root
    train_df = collect_train_rows(root, args.iteration)
    eval_df = collect_eval_rows(root, args.iteration)

    train_path = PROJECT_ROOT / args.train_csv
    eval_path = PROJECT_ROOT / args.eval_csv
    ensure_dir(train_path.parent)
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    mean_df = export_mean_table(eval_df, PROJECT_ROOT / args.mean_csv)

    fig_dir = PROJECT_ROOT / args.figures
    plot_metric_mean(eval_df, "psnr", "Held-out PSNR", fig_dir / "3dgs_multiscene_psnr_vs_views.png")
    plot_metric_mean(eval_df, "ssim", "Held-out SSIM", fig_dir / "3dgs_multiscene_ssim_vs_views.png")
    plot_metric_mean(eval_df, "lpips", "Held-out LPIPS", fig_dir / "3dgs_multiscene_lpips_vs_views.png")

    print(f"Saved train rows: {len(train_df)} -> {train_path}")
    print(f"Saved eval rows: {len(eval_df)} -> {eval_path}")
    print(f"Saved mean table: {PROJECT_ROOT / args.mean_csv}")
    print(f"Saved figures to {fig_dir}")
    print(mean_df.to_string(index=False))


if __name__ == "__main__":
    main()
