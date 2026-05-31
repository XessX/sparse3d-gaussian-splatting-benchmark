import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def load_per_view(eval_csv: Path) -> pd.DataFrame:
    eval_df = pd.read_csv(eval_csv)
    frames = []
    for _, row in eval_df.iterrows():
        per_view_path = Path(row["per_view_csv"])
        if not per_view_path.exists():
            continue
        per_view = pd.read_csv(per_view_path)
        per_view["scene"] = row["scene"]
        per_view["views"] = int(row["views"])
        per_view["selection"] = row["selection"]
        per_view["split"] = row["split"]
        frames.append(per_view)
    if not frames:
        raise FileNotFoundError(f"No per-view metric CSVs found from {eval_csv}")
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return tuple(np.percentile(means, [2.5, 97.5]))


def wilcoxon_p(values: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    if len(values) < 2 or np.allclose(values, 0):
        return None
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return None


def paired_rows(per_view: pd.DataFrame, target: str, comparator: str, views: int | None, metric: str) -> pd.DataFrame:
    df = per_view if views is None else per_view[per_view["views"] == views]
    target_df = df[df["selection"] == target][["scene", "views", "image", metric]].rename(columns={metric: "target"})
    comp_df = df[df["selection"] == comparator][["scene", "views", "image", metric]].rename(columns={metric: "comparator"})
    return target_df.merge(comp_df, on=["scene", "views", "image"], how="inner").dropna(subset=["target", "comparator"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", default="results/csv/3dgs_multiscene_eval_3000.csv")
    parser.add_argument("--target", default="quality_image_diverse")
    parser.add_argument("--comparators", nargs="+", default=["quality", "quality_diverse", "random", "uniform"])
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    per_view = load_per_view(PROJECT_ROOT / args.eval_csv)
    view_groups: list[int | None] = [None, *sorted(per_view["views"].dropna().astype(int).unique().tolist())]
    rows = []
    for views in view_groups:
        for comparator in args.comparators:
            for metric in ["psnr", "ssim", "lpips"]:
                paired = paired_rows(per_view, args.target, comparator, views, metric)
                label = "all" if views is None else str(views)
                if paired.empty:
                    rows.append(
                        {
                            "views": label,
                            "target": args.target,
                            "comparator": comparator,
                            "metric": metric,
                            "n_pairs": 0,
                            "mean_diff_target_minus_comparator": np.nan,
                            "bootstrap_ci_low": np.nan,
                            "bootstrap_ci_high": np.nan,
                            "wilcoxon_p": np.nan,
                            "interpretation": "TODO: missing paired evaluated rows",
                        }
                    )
                    continue
                diff = (paired["target"] - paired["comparator"]).to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_ci(diff, args.bootstrap_iterations, args.seed)
                p_value = wilcoxon_p(diff)
                if metric == "lpips":
                    direction = "lower is better; negative diff favors target"
                else:
                    direction = "higher is better; positive diff favors target"
                rows.append(
                    {
                        "views": label,
                        "target": args.target,
                        "comparator": comparator,
                        "metric": metric,
                        "n_pairs": len(diff),
                        "mean_diff_target_minus_comparator": float(np.mean(diff)),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "wilcoxon_p": p_value,
                        "interpretation": direction,
                    }
                )

    out = PROJECT_ROOT / "results" / "csv" / "statistical_tests.csv"
    ensure_dir(out.parent)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)

    table_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "tables")
    compact = result[result["views"].eq("all")].copy()
    columns = [
        "target",
        "comparator",
        "metric",
        "n_pairs",
        "mean_diff_target_minus_comparator",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "wilcoxon_p",
    ]
    display = compact[columns].copy()
    for col in ["mean_diff_target_minus_comparator", "bootstrap_ci_low", "bootstrap_ci_high", "wilcoxon_p"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in display.iterrows()]
    (table_dir / "table_statistical_tests.md").write_text(
        "# Table: Paired statistical tests for quality-image-diverse selection\n\n"
        "Differences are target minus comparator. For PSNR/SSIM, positive favors the target; for LPIPS, negative favors the target.\n\n"
        + "\n".join([header, divider, *body])
        + "\n",
        encoding="utf-8",
    )
    result.to_csv(table_dir / "table_statistical_tests.csv", index=False)
    print(f"Saved statistical tests to {out}")
    print(f"Saved manuscript table to {table_dir / 'table_statistical_tests.md'}")


if __name__ == "__main__":
    main()
