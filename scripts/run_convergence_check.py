import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rendered_images import evaluate
from materialize_sparse_scene import materialize_sparse_scene
from run_3dgs_sparse import parse_train_metrics
from src.utils.io import ensure_dir, save_json


DEFAULT_SCENES = ["tandt/truck", "db/playroom"]
DEFAULT_SPLITS = [
    "5views_uniform",
    "5views_random",
    "5views_quality_image_diverse",
    "20views_uniform",
]


def parse_views(split: str) -> int | None:
    match = re.match(r"(\d+)views_", split)
    return int(match.group(1)) if match else None


def existing_sparse_eval(scene: str, split: str, iteration: int) -> dict | None:
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
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def convergence_model_path(scene: str, split: str, iteration: int) -> Path:
    return PROJECT_ROOT / "results" / "method_outputs" / scene / "3dgs_convergence" / f"{split}_{iteration}"


def convergence_eval(scene: str, split: str, iteration: int) -> dict | None:
    path = convergence_model_path(scene, split, iteration) / "test" / f"ours_{iteration}" / "metrics_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_train(scene: str, split: str, iteration: int, resolution: int) -> dict:
    materialized = materialize_sparse_scene(scene, split, "datasets/convergence_scenes")
    source_path = Path(materialized["output_dir"])
    model_path = convergence_model_path(scene, split, iteration)
    if model_path.exists():
        shutil.rmtree(model_path)
    ensure_dir(model_path)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "external_methods" / "gaussian-splatting" / "train.py"),
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--iterations",
        str(iteration),
        "--save_iterations",
        str(iteration),
        "--test_iterations",
        str(iteration),
        "--disable_viewer",
        "--resolution",
        str(resolution),
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    runtime = time.perf_counter() - start
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (model_path / "train_console.log").write_text(output, encoding="utf-8")
    metrics = parse_train_metrics(output)
    report = {
        "scene": scene,
        "split": split,
        "iteration": iteration,
        "resolution": resolution,
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "runtime_sec": runtime,
        "train_psnr": metrics.get("train_psnr"),
        "train_l1": metrics.get("train_l1"),
        "source_path": str(source_path.relative_to(PROJECT_ROOT)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }
    save_json(report, model_path / "run_report.json")
    if proc.returncode != 0:
        raise RuntimeError(f"Convergence training failed for {scene} {split} {iteration}")
    return report


def run_render(scene: str, split: str, iteration: int, resolution: int) -> dict:
    materialized = materialize_sparse_scene(
        scene,
        split,
        "datasets/convergence_eval_scenes",
        f"datasets/processed/{scene}/eval_test_views.json",
    )
    source_path = Path(materialized["output_dir"])
    model_path = convergence_model_path(scene, split, iteration)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "external_methods" / "gaussian-splatting" / "render.py"),
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--iteration",
        str(iteration),
        "--skip_train",
        "--eval",
        "--resolution",
        str(resolution),
    ]
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (model_path / "render_console.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Convergence render failed for {scene} {split} {iteration}")
    return evaluate(model_path, f"ours_{iteration}")


def collect_rows(args) -> pd.DataFrame:
    rows = []
    for scene in args.scenes:
        for split in args.splits:
            for iteration in args.iterations:
                source = "existing_3dgs_sparse" if iteration == 3000 else "3dgs_convergence"
                metrics = existing_sparse_eval(scene, split, iteration) if iteration == 3000 else convergence_eval(scene, split, iteration)
                train_report = None
                status = "evaluated" if metrics else "missing"
                if metrics is None and args.run_missing:
                    train_report = run_train(scene, split, iteration, args.resolution)
                    metrics = run_render(scene, split, iteration, args.resolution)
                    status = "evaluated"
                rows.append(
                    {
                        "scene": scene,
                        "split": split,
                        "views": parse_views(split),
                        "iteration": iteration,
                        "source": source,
                        "status": status,
                        "psnr": metrics.get("psnr") if metrics else None,
                        "ssim": metrics.get("ssim") if metrics else None,
                        "lpips": metrics.get("lpips") if metrics else None,
                        "num_test_images": metrics.get("num_images") if metrics else None,
                        "train_runtime_sec": train_report.get("runtime_sec") if train_report else None,
                        "todo": "" if metrics else "Run with --run-missing to train/render this convergence case.",
                    }
                )
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, output: Path) -> None:
    table = df.copy()
    for column in ["psnr", "ssim", "lpips", "train_runtime_sec"]:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    columns = ["scene", "split", "iteration", "status", "psnr", "ssim", "lpips", "train_runtime_sec", "todo"]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in table.iterrows()]
    output.write_text("# Table: 3DGS convergence check\n\n" + "\n".join([header, divider, *rows]) + "\n", encoding="utf-8")


def plot_convergence(df: pd.DataFrame, output: Path) -> None:
    plot_df = df[df["status"] == "evaluated"].dropna(subset=["psnr"]).copy()
    ensure_dir(output.parent)
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No convergence runs evaluated yet", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output, dpi=300)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (scene, split), group in plot_df.groupby(["scene", "split"]):
        group = group.sort_values("iteration")
        ax.plot(group["iteration"], group["psnr"], marker="o", linewidth=2, label=f"{scene} / {split}")
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Held-out PSNR")
    ax.set_title("3DGS convergence check")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--iterations", nargs="+", type=int, default=[3000, 7000])
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--csv", default="results/csv/convergence_check.csv")
    args = parser.parse_args()

    df = collect_rows(args)
    csv_path = PROJECT_ROOT / args.csv
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)

    table_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "tables")
    figure_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "figures")
    write_markdown_table(df, table_dir / "table_convergence_check.md")
    df.to_csv(table_dir / "table_convergence_check.csv", index=False)
    plot_convergence(df, figure_dir / "3dgs_convergence_check.png")
    plot_convergence(df, PROJECT_ROOT / "results" / "figures" / "3dgs_convergence_check.png")
    print(f"Saved convergence rows to {csv_path}")
    print(f"Saved manuscript table and figure.")


if __name__ == "__main__":
    main()
