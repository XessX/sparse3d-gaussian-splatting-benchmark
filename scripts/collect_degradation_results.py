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
    return df


def load_clean_baseline(path: Path, splits: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"].isin(splits)].copy()
    df["condition"] = "clean"
    df["split_name"] = df["split"]
    df["eval_psnr"] = df["psnr"]
    df["eval_ssim"] = df["ssim"]
    df["eval_lpips"] = df["lpips"]
    keep = ["scene", "condition", "split_name", "num_test_images", "eval_psnr", "eval_ssim", "eval_lpips"]
    return df[keep]


def summarize(degraded_path: Path, clean_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    degraded = pd.read_csv(degraded_path)
    splits = set(degraded["split_name"].unique())
    clean = load_clean_baseline(clean_path, splits)
    combined = pd.concat(
        [
            degraded[["scene", "condition", "split_name", "num_test_images", "eval_psnr", "eval_ssim", "eval_lpips"]],
            clean,
        ],
        ignore_index=True,
    )
    combined = enrich(combined)
    combined = combined.sort_values(["condition", "views", "selection", "scene"])

    mean = (
        combined.groupby(["condition", "views", "selection"], as_index=False)
        .agg(
            scenes=("scene", "nunique"),
            psnr_mean=("eval_psnr", "mean"),
            psnr_std=("eval_psnr", "std"),
            ssim_mean=("eval_ssim", "mean"),
            ssim_std=("eval_ssim", "std"),
            lpips_mean=("eval_lpips", "mean"),
            lpips_std=("eval_lpips", "std"),
        )
        .sort_values(["condition", "views", "selection"])
    )

    ensure_dir(output_dir)
    combined.to_csv(output_dir / "3dgs_degradation_eval_with_clean.csv", index=False)
    mean.to_csv(output_dir / "3dgs_degradation_eval_mean.csv", index=False)
    return combined, mean


def plot_condition_bars(mean: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    plot_df = mean[mean["views"] == 5].copy()
    order = ["clean", "jpeg_q20", "jpeg_q40", "blur_k15", "lowres_50", "under_0.6", "under_0.4", "over_1.4"]
    labels = {
        "clean": "Clean",
        "jpeg_q20": "JPEG Q20",
        "jpeg_q40": "JPEG Q40",
        "blur_k15": "Motion blur",
        "lowres_50": "Low-res 50%",
        "under_0.6": "Under 0.6",
        "under_0.4": "Under 0.4",
        "over_1.4": "Over 1.4",
    }
    selection_labels = {
        "quality": "quality",
        "quality_diverse": "quality+diverse",
        "random": "random",
        "uniform": "uniform",
    }
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=order, ordered=True)
    plot_df["condition_label"] = plot_df["condition"].map(labels)
    plot_df["selection_label"] = plot_df["selection"].map(selection_labels).fillna(plot_df["selection"])
    for metric, ylabel in [
        ("psnr_mean", "Held-out PSNR"),
        ("ssim_mean", "Held-out SSIM"),
        ("lpips_mean", "Held-out LPIPS"),
    ]:
        pivot = plot_df.pivot(index="condition_label", columns="selection_label", values=metric)
        ax = pivot.plot(kind="bar", figsize=(8, 4.8))
        ax.set_xlabel("Training image condition")
        ax.set_ylabel(ylabel)
        ax.set_title(f"5-view degradation robustness ({ylabel})")
        ax.legend(title="Selection")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"3dgs_degradation_5views_{metric}.png", dpi=300)
        plt.close()


def plot_drop_from_clean(mean: pd.DataFrame, output_dir: Path) -> None:
    plot_df = mean[mean["views"] == 5].copy()
    labels = {
        "jpeg_q20": "JPEG Q20",
        "jpeg_q40": "JPEG Q40",
        "blur_k15": "Motion blur",
        "lowres_50": "Low-res 50%",
        "under_0.6": "Under 0.6",
        "under_0.4": "Under 0.4",
        "over_1.4": "Over 1.4",
    }
    selection_labels = {
        "quality": "quality",
        "quality_diverse": "quality+diverse",
        "random": "random",
        "uniform": "uniform",
    }
    clean = plot_df[plot_df["condition"] == "clean"][["selection", "psnr_mean"]].rename(
        columns={"psnr_mean": "clean_psnr"}
    )
    merged = plot_df.merge(clean, on="selection", how="left")
    merged["psnr_drop"] = merged["clean_psnr"] - merged["psnr_mean"]
    merged = merged[merged["condition"] != "clean"]
    merged["condition_label"] = merged["condition"].map(labels)
    merged["selection_label"] = merged["selection"].map(selection_labels).fillna(merged["selection"])
    pivot = merged.pivot(index="condition_label", columns="selection_label", values="psnr_drop")
    ax = pivot.plot(kind="bar", figsize=(8, 4.8))
    ax.set_xlabel("Training image degradation")
    ax.set_ylabel("PSNR drop from clean")
    ax.set_title("5-view degradation sensitivity")
    ax.legend(title="Selection")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "3dgs_degradation_5views_psnr_drop.png", dpi=300)
    plt.close()


def save_drop_table(mean: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    drop_df = mean[mean["views"] == 5].copy()
    clean = drop_df[drop_df["condition"] == "clean"][["selection", "psnr_mean", "ssim_mean", "lpips_mean"]].rename(
        columns={
            "psnr_mean": "clean_psnr_mean",
            "ssim_mean": "clean_ssim_mean",
            "lpips_mean": "clean_lpips_mean",
        }
    )
    merged = drop_df.merge(clean, on="selection", how="left")
    merged["psnr_drop_from_clean"] = merged["clean_psnr_mean"] - merged["psnr_mean"]
    merged["ssim_drop_from_clean"] = merged["clean_ssim_mean"] - merged["ssim_mean"]
    merged["lpips_increase_from_clean"] = merged["lpips_mean"] - merged["clean_lpips_mean"]
    merged = merged.sort_values(["condition", "selection"])
    ensure_dir(output_dir)
    merged.to_csv(output_dir / "3dgs_degradation_psnr_drop.csv", index=False)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degraded", default="results/csv/3dgs_degradation_runs_3000.csv")
    parser.add_argument("--clean", default="results/csv/3dgs_multiscene_eval_3000.csv")
    parser.add_argument("--output", default="results/csv/degradation")
    parser.add_argument("--figures", default="results/figures/3dgs_degradation")
    args = parser.parse_args()

    combined, mean = summarize(PROJECT_ROOT / args.degraded, PROJECT_ROOT / args.clean, PROJECT_ROOT / args.output)
    drop = save_drop_table(mean, PROJECT_ROOT / args.output)
    fig_dir = PROJECT_ROOT / args.figures
    plot_condition_bars(mean, fig_dir)
    plot_drop_from_clean(mean, fig_dir)
    print(f"Saved combined rows: {len(combined)}")
    print(f"Saved mean rows: {len(mean)}")
    print(f"Saved drop rows: {len(drop)}")
    print(mean.to_string(index=False))


if __name__ == "__main__":
    main()
