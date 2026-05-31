import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rendered_images import evaluate
from materialize_sparse_scene import materialize_sparse_scene
from src.utils.io import ensure_dir, save_json


DEFAULT_SPLITS = [
    "2views_uniform",
    "2views_random",
    "2views_quality",
    "2views_quality_diverse",
    "3views_uniform",
    "3views_random",
    "3views_quality",
    "3views_quality_diverse",
    "5views_uniform",
    "5views_random",
    "5views_quality",
    "5views_quality_diverse",
    "8views_uniform",
    "8views_random",
    "8views_quality",
    "8views_quality_diverse",
    "12views_uniform",
    "12views_random",
    "12views_quality",
    "12views_quality_diverse",
    "20views_uniform",
    "20views_random",
    "20views_quality",
    "20views_quality_diverse",
]


def render_split(scene: str, split: str, iteration: int, resolution: int, skip_existing: bool) -> dict:
    materialized = materialize_sparse_scene(
        scene,
        split,
        "datasets/eval_scenes",
        f"datasets/processed/{scene}/eval_test_views.json",
    )
    source_path = Path(materialized["output_dir"])
    model_path = PROJECT_ROOT / "results" / "method_outputs" / scene / "3dgs_sparse" / split
    render_dir = model_path / "test" / f"ours_{iteration}" / "renders"
    if skip_existing and render_dir.exists() and any(render_dir.glob("*.png")):
        rendered = False
        render_returncode = 0
    else:
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
        render_returncode = proc.returncode
        rendered = True
        log_path = model_path / "render_console.log"
        log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Render failed for {split}. See {log_path}")

    summary = evaluate(model_path, f"ours_{iteration}")
    report = {
        "scene": scene,
        "split": split,
        "iteration": iteration,
        "resolution": resolution,
        "rendered": rendered,
        "render_returncode": render_returncode,
        "eval": summary,
    }
    save_json(report, model_path / "render_eval_report.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="tandt/train")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--report", default="results/logs/3dgs_render_eval_report.json")
    args = parser.parse_args()

    reports = [
        render_split(args.scene, split, args.iteration, args.resolution, args.skip_existing)
        for split in args.splits
    ]
    report_path = PROJECT_ROOT / args.report
    ensure_dir(report_path.parent)
    save_json({"runs": reports}, report_path)
    print(json.dumps(reports, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
