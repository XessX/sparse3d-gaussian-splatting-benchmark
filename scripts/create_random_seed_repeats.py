import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import list_images
from src.utils.io import ensure_dir


def scene_leaf(scene: str) -> str:
    return Path(scene).name


def load_eval_test_images(scene: str) -> set[str]:
    path = PROJECT_ROOT / "datasets" / "processed" / scene / "eval_test_views.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("test_images", []))


def load_metrics(scene: str, split: str, iteration: int) -> dict:
    path = (
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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_md_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()
    for col in table.select_dtypes(include=["float"]).columns:
        table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in table.iterrows()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"])
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 5, 8, 12, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iteration", type=int, default=3000)
    args = parser.parse_args()

    rows = []
    for scene in args.scenes:
        leaf = scene_leaf(scene)
        image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
        split_path = PROJECT_ROOT / "datasets" / "processed" / scene / "splits" / f"{leaf}_splits.json"
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        test_images = load_eval_test_images(scene)
        candidates = [path.name for path in list_images(image_dir) if path.name not in test_images]

        for seed in args.seeds:
            rng = random.Random(seed)
            for n in args.views:
                if n > len(candidates):
                    raise ValueError(f"{scene} has only {len(candidates)} candidates after test exclusion")
                split = f"{n}views_random_seed{seed}"
                selected = sorted(rng.sample(candidates, n))
                split_data["splits"][split] = selected
                metrics = load_metrics(scene, split, args.iteration)
                rows.append(
                    {
                        "scene": scene,
                        "views": n,
                        "seed": seed,
                        "split": split,
                        "status": "evaluated" if metrics else "prepared_not_run",
                        "selected_images": json.dumps(selected),
                        "excluded_test_images": len(test_images),
                        "psnr": metrics.get("psnr"),
                        "ssim": metrics.get("ssim"),
                        "lpips": metrics.get("lpips"),
                        "num_test_images": metrics.get("num_images"),
                        "metrics_summary": str(metrics.get("model_path", "")) if metrics else "",
                        "todo": "" if metrics else "Run 3DGS train/render/evaluate for this split.",
                    }
                )

        split_data["random_seed_repeat_selection"] = {
            "seeds": args.seeds,
            "views": args.views,
            "excluded_eval_test_views": len(test_images),
            "notes": "Repeated random splits are generated after excluding held-out eval views. Rows marked evaluated have corresponding 3DGS render metrics.",
        }
        split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "csv" / "random_seed_repeats.csv"
    ensure_dir(out.parent)
    df.to_csv(out, index=False)

    summary_rows = []
    for views, group in df.groupby("views"):
        evaluated = group[group["status"] == "evaluated"].copy()
        planned_runs = int(len(group))
        evaluated_runs = int(len(evaluated))
        row = {
            "views": views,
            "scenes": int(group["scene"].nunique()),
            "planned_runs": planned_runs,
            "evaluated_runs": evaluated_runs,
            "psnr_mean": evaluated["psnr"].mean() if evaluated_runs else pd.NA,
            "psnr_std": evaluated["psnr"].std() if evaluated_runs else pd.NA,
            "ssim_mean": evaluated["ssim"].mean() if evaluated_runs else pd.NA,
            "ssim_std": evaluated["ssim"].std() if evaluated_runs else pd.NA,
            "lpips_mean": evaluated["lpips"].mean() if evaluated_runs else pd.NA,
            "lpips_std": evaluated["lpips"].std() if evaluated_runs else pd.NA,
            "status": "complete" if evaluated_runs == planned_runs else ("prepared_only" if evaluated_runs == 0 else "partial"),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("views")

    summary_path = PROJECT_ROOT / "results" / "csv" / "random_seed_repeats_summary.csv"
    summary.to_csv(summary_path, index=False)

    table_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "tables")
    md = (
        "# Table: Repeated random-seed sparse-view baseline\n\n"
        "Random splits for seeds 0-4 were generated after excluding the held-out test views. "
        "Rows marked complete have measured 3DGS held-out render metrics for every planned scene/seed pair.\n\n"
        + format_md_table(summary, ["views", "scenes", "planned_runs", "evaluated_runs", "psnr_mean", "psnr_std", "status"])
        + "\n"
    )
    (table_dir / "table_random_seed_repeats.md").write_text(md, encoding="utf-8")
    summary.to_csv(table_dir / "table_random_seed_repeats.csv", index=False)

    print(f"Saved random repeat rows to {out}")
    print(f"Saved random repeat summary to {summary_path}")
    print(f"Saved manuscript table to {table_dir / 'table_random_seed_repeats.md'}")


if __name__ == "__main__":
    main()
