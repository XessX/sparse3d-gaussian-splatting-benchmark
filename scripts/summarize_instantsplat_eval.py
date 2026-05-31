import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


SELECTION_ORDER = ["uniform", "random", "quality", "quality_diverse"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["selection"] = df["split"].str.replace(r"^\d+views_", "", regex=True)
    df["success"] = df["status"].eq("success")
    for column in [
        "views",
        "train_iterations",
        "num_test_images",
        "init_geo_sec",
        "train_sec",
        "render_sec",
        "psnr",
        "ssim",
        "lpips",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["runtime_sec"] = df[["init_geo_sec", "train_sec", "render_sec"]].sum(axis=1)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby("selection", as_index=False)
        .agg(
            scenes=("scene", "nunique"),
            runs=("scene", "size"),
            success_runs=("success", "sum"),
            success_rate=("success", "mean"),
        )
    )
    success = df[df["success"]].copy()
    metrics = (
        success.groupby("selection", as_index=False)
        .agg(
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            lpips_mean=("lpips", "mean"),
            lpips_std=("lpips", "std"),
            init_geo_sec_mean=("init_geo_sec", "mean"),
            train_sec_mean=("train_sec", "mean"),
            render_sec_mean=("render_sec", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
    )
    summary = base.merge(metrics, on="selection", how="left")
    summary["selection_order"] = summary["selection"].map(
        {selection: idx for idx, selection in enumerate(SELECTION_ORDER)}
    )
    return summary.sort_values(["selection_order", "selection"]).drop(columns=["selection_order"])


def save_bar(summary: pd.DataFrame, value_col: str, ylabel: str, title: str, output_name: str) -> None:
    figure_dir = ensure_dir(PROJECT_ROOT / "results" / "figures" / "foundation")
    manuscript_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "figures")
    plot_df = summary.copy()
    plot_df["selection_order"] = plot_df["selection"].map(
        {selection: idx for idx, selection in enumerate(SELECTION_ORDER)}
    )
    plot_df = plot_df.sort_values(["selection_order", "selection"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = [value.replace("_", "\n") for value in plot_df["selection"]]
    ax.bar(labels, plot_df[value_col], color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    result_path = figure_dir / output_name
    manuscript_path = manuscript_dir / output_name
    fig.savefig(result_path, dpi=300)
    fig.savefig(manuscript_path, dpi=300)
    plt.close(fig)


def resize_to_cell(image: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(cell_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", cell_size, "white")
    left = (cell_size[0] - image.width) // 2
    top = (cell_size[1] - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def image_path_for(row: pd.Series, subdir: str, image_name: str) -> Path:
    return Path(row["output_dir"]) / "test" / f"ours_{int(row['train_iterations'])}" / subdir / image_name


def save_qualitative_grid(df: pd.DataFrame, scene: str, output_name: str, max_rows: int = 4) -> Path | None:
    scene_df = df[(df["scene"] == scene) & (df["success"])].copy()
    if scene_df.empty:
        return None

    rows_by_selection = {
        row["selection"]: row
        for _, row in scene_df.iterrows()
        if row["selection"] in SELECTION_ORDER
    }
    if not rows_by_selection:
        return None

    first_row = next(iter(rows_by_selection.values()))
    gt_dir = Path(first_row["output_dir"]) / "test" / f"ours_{int(first_row['train_iterations'])}" / "gt"
    image_names = sorted(path.name for path in gt_dir.glob("*.png"))[:max_rows]
    if not image_names:
        return None

    selections = [selection for selection in SELECTION_ORDER if selection in rows_by_selection]
    columns = ["ground truth", *selections]
    cell_size = (220, 124)
    label_height = 28
    grid = Image.new(
        "RGB",
        (len(columns) * cell_size[0], (len(image_names) + 1) * cell_size[1] + label_height),
        "white",
    )
    draw = ImageDraw.Draw(grid)

    for col_idx, label in enumerate(columns):
        draw.text((col_idx * cell_size[0] + 8, 8), label.replace("_", " "), fill="black")

    for row_idx, image_name in enumerate(image_names):
        y = label_height + row_idx * cell_size[1]
        gt_path = gt_dir / image_name
        if gt_path.exists():
            grid.paste(resize_to_cell(Image.open(gt_path), cell_size), (0, y))

        for col_idx, selection in enumerate(selections, start=1):
            render_path = image_path_for(rows_by_selection[selection], "renders", image_name)
            if render_path.exists():
                grid.paste(resize_to_cell(Image.open(render_path), cell_size), (col_idx * cell_size[0], y))

    out = PROJECT_ROOT / "results" / "qualitative" / output_name
    manuscript_out = PROJECT_ROOT / "manuscript" / "figures" / output_name
    ensure_dir(out.parent)
    ensure_dir(manuscript_out.parent)
    grid.save(out)
    shutil.copy2(out, manuscript_out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/csv/foundation/instantsplat_eval_5views.csv")
    parser.add_argument("--summary-csv", default="results/csv/foundation/instantsplat_eval_5views_mean.csv")
    parser.add_argument("--qualitative-scene", default="tandt/truck")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.csv
    df = normalize(pd.read_csv(input_path))
    summary = summarize(df)

    normalized_path = input_path.with_name(input_path.stem + "_normalized.csv")
    summary_path = PROJECT_ROOT / args.summary_csv
    ensure_dir(summary_path.parent)
    df.to_csv(normalized_path, index=False)
    summary.to_csv(summary_path, index=False)

    save_bar(
        summary,
        "psnr_mean",
        "PSNR",
        "InstantSplat 5-view held-out PSNR",
        "instantsplat_eval_5views_psnr.png",
    )
    save_bar(
        summary,
        "ssim_mean",
        "SSIM",
        "InstantSplat 5-view held-out SSIM",
        "instantsplat_eval_5views_ssim.png",
    )
    save_bar(
        summary,
        "lpips_mean",
        "LPIPS",
        "InstantSplat 5-view held-out LPIPS",
        "instantsplat_eval_5views_lpips.png",
    )
    save_bar(
        summary,
        "runtime_sec_mean",
        "Runtime (sec)",
        "InstantSplat 5-view end-to-end runtime",
        "instantsplat_eval_5views_runtime.png",
    )
    qualitative = save_qualitative_grid(
        df,
        scene=args.qualitative_scene,
        output_name="instantsplat_eval_5views_grid.png",
    )

    print(summary.to_string(index=False))
    print(f"Saved normalized rows to {normalized_path}")
    print(f"Saved summary to {summary_path}")
    if qualitative:
        print(f"Saved qualitative grid to {qualitative}")


if __name__ == "__main__":
    main()
