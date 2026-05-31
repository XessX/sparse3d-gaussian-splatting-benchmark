import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


FOUNDATION_DIR = PROJECT_ROOT / "results" / "csv" / "foundation"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "foundation"
MANUSCRIPT_FIGURE_DIR = PROJECT_ROOT / "manuscript" / "figures"

METHOD_5VIEW_FILES = [
    FOUNDATION_DIR / "vggt_pose_5views.csv",
    FOUNDATION_DIR / "dust3r_pose_5views.csv",
    FOUNDATION_DIR / "mast3r_pose_5views.csv",
]

SELECTION_ORDER = ["uniform", "random", "quality", "quality_diverse"]
METHOD_ORDER = ["VGGT", "DUSt3R", "MASt3R"]


def read_existing_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No foundation pose CSV files were found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def compute_runtime(row: pd.Series) -> float:
    runtime_columns = [
        "load_images_sec",
        "image_load_sec",
        "infer_sec",
        "inference_sec",
        "alignment_sec",
    ]
    total = 0.0
    for col in runtime_columns:
        value = row.get(col)
        if pd.notna(value):
            total += float(value)
    return total


def normalize_foundation_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "selection" not in df.columns:
        df["selection"] = df["split"].str.replace(r"^\d+views_", "", regex=True)
    df["runtime_sec"] = df.apply(compute_runtime, axis=1)
    df["success"] = df["status"].eq("success")
    df["normalized_camera_center_rmse"] = pd.to_numeric(
        df["normalized_camera_center_rmse"], errors="coerce"
    )
    df["peak_gpu_memory_mb"] = pd.to_numeric(df["peak_gpu_memory_mb"], errors="coerce")
    return df


def summarize_5view(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = df[df["success"]].copy()
    by_selection = (
        success.groupby(["method", "selection"], as_index=False)
        .agg(
            scenes=("scene", "nunique"),
            runs=("scene", "size"),
            camera_rmse_mean=("normalized_camera_center_rmse", "mean"),
            camera_rmse_std=("normalized_camera_center_rmse", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean"),
        )
        .sort_values(["method", "selection"])
    )
    overall = (
        success.groupby("method", as_index=False)
        .agg(
            scenes=("scene", "nunique"),
            runs=("scene", "size"),
            camera_rmse_mean=("normalized_camera_center_rmse", "mean"),
            camera_rmse_std=("normalized_camera_center_rmse", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean"),
        )
        .sort_values("camera_rmse_mean")
    )
    return by_selection, overall


def summarize_vggt_sparse() -> pd.DataFrame | None:
    path = FOUNDATION_DIR / "vggt_pose_sparse_all.csv"
    if not path.exists():
        return None
    df = normalize_foundation_rows(pd.read_csv(path))
    success = df[df["success"]].copy()
    summary = (
        success.groupby(["views", "selection"], as_index=False)
        .agg(
            scenes=("scene", "nunique"),
            runs=("scene", "size"),
            camera_rmse_mean=("normalized_camera_center_rmse", "mean"),
            camera_rmse_std=("normalized_camera_center_rmse", "std"),
            infer_sec_mean=("infer_sec", "mean"),
            peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean"),
        )
        .sort_values(["views", "selection"])
    )
    return summary


def save_bar_plot(
    summary: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> None:
    ensure_dir(FIGURE_DIR)
    ensure_dir(MANUSCRIPT_FIGURE_DIR)

    selections = [s for s in SELECTION_ORDER if s in set(summary["selection"])]
    methods = [m for m in METHOD_ORDER if m in set(summary["method"])]
    x = np.arange(len(selections))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, method in enumerate(methods):
        values = []
        for selection in selections:
            match = summary[
                (summary["method"] == method) & (summary["selection"] == selection)
            ]
            values.append(float(match[value_col].iloc[0]) if not match.empty else np.nan)
        offset = (index - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in selections])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = FIGURE_DIR / output_name
    manuscript_out = MANUSCRIPT_FIGURE_DIR / output_name
    fig.savefig(out, dpi=300)
    fig.savefig(manuscript_out, dpi=300)
    plt.close(fig)


def save_vggt_sparse_plot(summary: pd.DataFrame) -> None:
    ensure_dir(FIGURE_DIR)
    ensure_dir(MANUSCRIPT_FIGURE_DIR)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for selection in SELECTION_ORDER:
        group = summary[summary["selection"] == selection].sort_values("views")
        if group.empty:
            continue
        ax.plot(
            group["views"],
            group["camera_rmse_mean"],
            marker="o",
            linewidth=2,
            label=selection.replace("_", " "),
        )
    ax.set_xlabel("Input views")
    ax.set_ylabel("Normalized camera-center RMSE")
    ax.set_title("VGGT pose error across sparse-view settings")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = FIGURE_DIR / "vggt_pose_sparse_rmse_vs_views.png"
    manuscript_out = MANUSCRIPT_FIGURE_DIR / "vggt_pose_sparse_rmse_vs_views.png"
    fig.savefig(out, dpi=300)
    fig.savefig(manuscript_out, dpi=300)
    plt.close(fig)


def main():
    ensure_dir(FOUNDATION_DIR)
    combined = normalize_foundation_rows(read_existing_csvs(METHOD_5VIEW_FILES))
    summary, overall = summarize_5view(combined)

    combined_path = FOUNDATION_DIR / "foundation_pose_5views_combined.csv"
    summary_path = FOUNDATION_DIR / "foundation_pose_5views_mean.csv"
    overall_path = FOUNDATION_DIR / "foundation_pose_5views_overall.csv"
    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    overall.to_csv(overall_path, index=False)

    save_bar_plot(
        summary,
        "camera_rmse_mean",
        "Normalized camera-center RMSE",
        "5-view pose accuracy for foundation reconstruction models",
        "foundation_pose_5views_rmse.png",
    )
    save_bar_plot(
        summary,
        "runtime_sec_mean",
        "Runtime per split (sec)",
        "5-view foundation-model runtime",
        "foundation_pose_5views_runtime.png",
    )

    vggt_sparse = summarize_vggt_sparse()
    if vggt_sparse is not None:
        sparse_path = FOUNDATION_DIR / "vggt_pose_sparse_mean.csv"
        vggt_sparse.to_csv(sparse_path, index=False)
        save_vggt_sparse_plot(vggt_sparse)
        print(f"Saved VGGT sparse summary to {sparse_path}")

    print(f"Saved combined foundation pose rows to {combined_path}")
    print(f"Saved foundation pose summary to {summary_path}")
    print(f"Saved foundation pose overall summary to {overall_path}")
    print(f"Saved foundation figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
