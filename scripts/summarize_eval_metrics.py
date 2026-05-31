import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def parse_split(model_path: Path) -> tuple[int | None, str]:
    name = model_path.name
    if "views_" not in name:
        return None, name
    view_part, selection = name.split("views_", 1)
    try:
        return int(view_part), selection
    except ValueError:
        return None, selection


def collect_eval_summaries(input_root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(input_root.glob("*/test/*/metrics_summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        model_path = Path(summary["model_path"])
        views, selection = parse_split(model_path)
        rows.append(
            {
                "scene": "tandt/train",
                "method": "colmap_3dgs",
                "split": model_path.name,
                "views": views,
                "selection": selection,
                "render_method": summary.get("method"),
                "num_test_images": summary.get("num_images"),
                "psnr": summary.get("psnr"),
                "ssim": summary.get("ssim"),
                "lpips": summary.get("lpips"),
                "metrics_summary": str(summary_path),
                "per_view_csv": summary.get("per_view_csv"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No metrics_summary.json files found under {input_root}")
    return pd.DataFrame(rows).sort_values(["views", "selection"])


def plot_eval_bars(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    metric_info = {
        "psnr": ("PSNR", "Higher is better"),
        "ssim": ("SSIM", "Higher is better"),
        "lpips": ("LPIPS", "Lower is better"),
    }
    for metric, (label, subtitle) in metric_info.items():
        plot_df = df.dropna(subset=[metric]).sort_values(["selection", "views"])
        plt.figure(figsize=(7, 4.5))
        for selection, group in plot_df.groupby("selection"):
            plt.plot(group["views"], group[metric], marker="o", linewidth=2, label=selection)
        plt.xlabel("Number of training views")
        plt.ylabel(label)
        plt.title(f"Held-out novel-view {label} across sparse-view settings\n{subtitle}")
        plt.legend(title="Selection")
        plt.tight_layout()
        plt.savefig(output_dir / f"3dgs_eval_{metric}_vs_views.png", dpi=300)
        plt.close()

        five_view = plot_df[plot_df["views"] == 5].sort_values(
            metric, ascending=(metric == "lpips")
        )
        if five_view.empty:
            continue
        plt.figure(figsize=(7, 4.5))
        plt.bar(five_view["selection"], five_view[metric])
        plt.xlabel("View selection strategy")
        plt.ylabel(label)
        plt.title(f"Held-out novel-view {label} for 5-view COLMAP + 3DGS\n{subtitle}")
        plt.tight_layout()
        plt.savefig(output_dir / f"3dgs_eval_5views_{metric}.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="results/method_outputs/tandt/train/3dgs_sparse",
        help="Root containing split folders with test/ours_*/metrics_summary.json files.",
    )
    parser.add_argument("--csv", default="results/csv/3dgs_eval_summary.csv")
    parser.add_argument("--figures", default="results/figures/3dgs_eval")
    args = parser.parse_args()

    df = collect_eval_summaries(PROJECT_ROOT / args.input_root)
    csv_path = PROJECT_ROOT / args.csv
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    plot_eval_bars(df, PROJECT_ROOT / args.figures)

    print(df[["split", "num_test_images", "psnr", "ssim", "lpips"]].to_string(index=False))
    print(f"Saved {csv_path}")
    print(f"Saved figures to {PROJECT_ROOT / args.figures}")


if __name__ == "__main__":
    main()
