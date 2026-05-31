import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def format_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()
    for col in table.select_dtypes(include=["float"]).columns:
        table[col] = table[col].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in table.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main():
    table_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "tables")
    eval_df = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "3dgs_eval_summary_clean.csv")
    train_df = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "3dgs_sparse_summary_3000_clean.csv")
    multiscene_eval = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "3dgs_multiscene_eval_3000.csv")
    multiscene_mean = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "3dgs_multiscene_eval_mean_3000.csv")
    multiscene_train = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "3dgs_multiscene_train_3000.csv")
    degradation_eval = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "degradation" / "3dgs_degradation_eval_with_clean.csv")
    degradation_mean = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "degradation" / "3dgs_degradation_eval_mean.csv")
    degradation_drop = pd.read_csv(PROJECT_ROOT / "results" / "csv" / "degradation" / "3dgs_degradation_psnr_drop.csv")
    foundation_dir = PROJECT_ROOT / "results" / "csv" / "foundation"
    foundation_5views = read_optional_csv(foundation_dir / "foundation_pose_5views_mean.csv")
    foundation_overall = read_optional_csv(foundation_dir / "foundation_pose_5views_overall.csv")
    vggt_sparse = read_optional_csv(foundation_dir / "vggt_pose_sparse_mean.csv")
    instantsplat_smoke = read_optional_csv(foundation_dir / "instantsplat_smoke.csv")
    instantsplat_eval = read_optional_csv(foundation_dir / "instantsplat_eval_5views.csv")
    instantsplat_eval_mean = read_optional_csv(foundation_dir / "instantsplat_eval_5views_mean.csv")

    eval_columns = ["split", "num_test_images", "psnr", "ssim", "lpips"]
    train_columns = ["split_name", "views", "selection", "train_psnr", "runtime_sec", "initial_points"]
    multiscene_mean_columns = [
        "views",
        "selection",
        "scenes",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "lpips_mean",
        "lpips_std",
    ]
    multiscene_eval_columns = ["scene", "split", "num_test_images", "psnr", "ssim", "lpips"]
    degradation_mean_columns = [
        "condition",
        "views",
        "selection",
        "scenes",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "lpips_mean",
        "lpips_std",
    ]
    degradation_eval_columns = [
        "scene",
        "condition",
        "split_name",
        "num_test_images",
        "eval_psnr",
        "eval_ssim",
        "eval_lpips",
    ]
    degradation_drop_columns = [
        "condition",
        "views",
        "selection",
        "scenes",
        "psnr_mean",
        "clean_psnr_mean",
        "psnr_drop_from_clean",
        "ssim_drop_from_clean",
        "lpips_increase_from_clean",
    ]
    foundation_columns = [
        "method",
        "selection",
        "scenes",
        "runs",
        "camera_rmse_mean",
        "camera_rmse_std",
        "runtime_sec_mean",
        "peak_gpu_memory_mb_mean",
    ]
    foundation_overall_columns = [
        "method",
        "scenes",
        "runs",
        "camera_rmse_mean",
        "camera_rmse_std",
        "runtime_sec_mean",
        "peak_gpu_memory_mb_mean",
    ]
    vggt_sparse_columns = [
        "views",
        "selection",
        "scenes",
        "runs",
        "camera_rmse_mean",
        "camera_rmse_std",
        "infer_sec_mean",
        "peak_gpu_memory_mb_mean",
    ]
    instantsplat_smoke_columns = [
        "scene",
        "split",
        "views",
        "status",
        "init_geo_sec",
        "train_iterations",
        "train_status",
        "train_sec",
        "saved_point_cloud_exists",
    ]
    instantsplat_eval_columns = [
        "scene",
        "split",
        "status",
        "num_test_images",
        "psnr",
        "ssim",
        "lpips",
        "init_geo_sec",
        "train_sec",
        "render_sec",
    ]
    instantsplat_eval_mean_columns = [
        "selection",
        "scenes",
        "runs",
        "success_rate",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "lpips_mean",
        "lpips_std",
        "runtime_sec_mean",
    ]

    eval_df.to_csv(table_dir / "table_3dgs_heldout_eval.csv", index=False)
    train_df.to_csv(table_dir / "table_3dgs_train_runtime.csv", index=False)
    multiscene_eval.to_csv(table_dir / "table_3dgs_multiscene_eval.csv", index=False)
    multiscene_mean.to_csv(table_dir / "table_3dgs_multiscene_eval_mean.csv", index=False)
    multiscene_train.to_csv(table_dir / "table_3dgs_multiscene_train.csv", index=False)
    degradation_eval.to_csv(table_dir / "table_3dgs_degradation_eval.csv", index=False)
    degradation_mean.to_csv(table_dir / "table_3dgs_degradation_eval_mean.csv", index=False)
    degradation_drop.to_csv(table_dir / "table_3dgs_degradation_psnr_drop.csv", index=False)
    if foundation_5views is not None:
        foundation_5views.to_csv(table_dir / "table_foundation_pose_5views_mean.csv", index=False)
    if foundation_overall is not None:
        foundation_overall.to_csv(table_dir / "table_foundation_pose_5views_overall.csv", index=False)
    if vggt_sparse is not None:
        vggt_sparse.to_csv(table_dir / "table_vggt_pose_sparse_mean.csv", index=False)
    if instantsplat_smoke is not None:
        instantsplat_smoke.to_csv(table_dir / "table_instantsplat_smoke.csv", index=False)
    if instantsplat_eval is not None:
        instantsplat_eval.to_csv(table_dir / "table_instantsplat_eval_5views.csv", index=False)
    if instantsplat_eval_mean is not None:
        instantsplat_eval_mean.to_csv(table_dir / "table_instantsplat_eval_5views_mean.csv", index=False)

    (table_dir / "table_3dgs_heldout_eval.md").write_text(
        "# Table: Held-out novel-view evaluation for COLMAP + 3DGS\n\n"
        + format_markdown_table(eval_df, eval_columns)
        + "\n",
        encoding="utf-8",
    )
    (table_dir / "table_3dgs_train_runtime.md").write_text(
        "# Table: Training metrics and runtime for COLMAP + 3DGS\n\n"
        + format_markdown_table(train_df, train_columns)
        + "\n",
        encoding="utf-8",
    )
    (table_dir / "table_3dgs_multiscene_eval_mean.md").write_text(
        "# Table: Multi-scene held-out evaluation mean and standard deviation\n\n"
        + format_markdown_table(multiscene_mean, multiscene_mean_columns)
        + "\n",
        encoding="utf-8",
    )
    (table_dir / "table_3dgs_multiscene_eval.md").write_text(
        "# Table: Per-scene held-out evaluation for COLMAP + 3DGS\n\n"
        + format_markdown_table(multiscene_eval, multiscene_eval_columns)
        + "\n",
        encoding="utf-8",
    )
    (table_dir / "table_3dgs_degradation_eval_mean.md").write_text(
        "# Table: Degradation robustness mean and standard deviation\n\n"
        + format_markdown_table(degradation_mean, degradation_mean_columns)
        + "\n",
        encoding="utf-8",
    )
    (table_dir / "table_3dgs_degradation_eval.md").write_text(
        "# Table: Per-scene degradation robustness for COLMAP + 3DGS\n\n"
        + format_markdown_table(degradation_eval, degradation_eval_columns)
        + "\n",
        encoding="utf-8",
    )
    (table_dir / "table_3dgs_degradation_psnr_drop.md").write_text(
        "# Table: Degradation change relative to clean 5-view baseline\n\n"
        + format_markdown_table(degradation_drop, degradation_drop_columns)
        + "\n",
        encoding="utf-8",
    )
    if foundation_5views is not None:
        (table_dir / "table_foundation_pose_5views_mean.md").write_text(
            "# Table: 5-view foundation-model camera-pose benchmark\n\n"
            + format_markdown_table(foundation_5views, foundation_columns)
            + "\n",
            encoding="utf-8",
        )
    if foundation_overall is not None:
        (table_dir / "table_foundation_pose_5views_overall.md").write_text(
            "# Table: Overall 5-view foundation-model camera-pose benchmark\n\n"
            + format_markdown_table(foundation_overall, foundation_overall_columns)
            + "\n",
            encoding="utf-8",
        )
    if vggt_sparse is not None:
        (table_dir / "table_vggt_pose_sparse_mean.md").write_text(
            "# Table: VGGT camera-pose benchmark across sparse-view settings\n\n"
            + format_markdown_table(vggt_sparse, vggt_sparse_columns)
            + "\n",
            encoding="utf-8",
        )
    if instantsplat_smoke is not None:
        (table_dir / "table_instantsplat_smoke.md").write_text(
            "# Table: InstantSplat executable smoke test\n\n"
            + format_markdown_table(instantsplat_smoke, instantsplat_smoke_columns)
            + "\n",
            encoding="utf-8",
        )
    if instantsplat_eval is not None:
        (table_dir / "table_instantsplat_eval_5views.md").write_text(
            "# Table: Per-scene InstantSplat 5-view held-out rendering evaluation\n\n"
            + format_markdown_table(instantsplat_eval, instantsplat_eval_columns)
            + "\n",
            encoding="utf-8",
        )
    if instantsplat_eval_mean is not None:
        (table_dir / "table_instantsplat_eval_5views_mean.md").write_text(
            "# Table: InstantSplat 5-view held-out rendering summary\n\n"
            + format_markdown_table(instantsplat_eval_mean, instantsplat_eval_mean_columns)
            + "\n",
            encoding="utf-8",
        )
    print(f"Saved manuscript tables to {table_dir}")


if __name__ == "__main__":
    main()
