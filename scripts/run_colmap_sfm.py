import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.executables import find_executable
from src.utils.io import ensure_dir


def run_command(command: list[str], log_path: Path) -> None:
    start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.write(f"elapsed_sec={elapsed:.3f}\n")
        if proc.stdout:
            log.write("\n[stdout]\n" + proc.stdout)
        if proc.stderr:
            log.write("\n[stderr]\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")


def run_colmap_sfm(scene: str, image_subdir: str, output_root: str, use_gpu: bool) -> dict:
    colmap = find_executable("colmap")
    if not colmap:
        raise FileNotFoundError("COLMAP executable not found. Run scripts/check_environment.py.")

    image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / image_subdir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir = ensure_dir(PROJECT_ROOT / output_root / scene / "colmap_sfm")
    sparse_dir = ensure_dir(output_dir / "sparse")
    database_path = output_dir / "database.db"
    log_path = output_dir / "colmap_sfm.log"
    if log_path.exists():
        log_path.unlink()

    gpu_flag = "1" if use_gpu else "0"
    total_start = time.perf_counter()

    run_command(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
            "--FeatureExtraction.use_gpu",
            gpu_flag,
        ],
        log_path,
    )
    run_command(
        [
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            gpu_flag,
        ],
        log_path,
    )
    run_command(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_global_function_tolerance",
            "0.000001",
        ],
        log_path,
    )

    models = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
    status = "success" if models else "no_model"
    report = {
        "scene": scene,
        "method": "colmap_sfm",
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "database_path": str(database_path),
        "sparse_dir": str(sparse_dir),
        "num_models": len(models),
        "status": status,
        "runtime_sec": time.perf_counter() - total_start,
        "colmap_executable": colmap,
        "use_gpu": use_gpu,
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--image-subdir", default="images")
    parser.add_argument("--output-root", default="results/method_outputs")
    parser.add_argument("--cpu", action="store_true", help="Disable COLMAP GPU feature extraction/matching")
    args = parser.parse_args()

    report = run_colmap_sfm(
        scene=args.scene,
        image_subdir=args.image_subdir,
        output_root=args.output_root,
        use_gpu=not args.cpu,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
