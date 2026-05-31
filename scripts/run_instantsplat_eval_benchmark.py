import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rendered_images import evaluate
from materialize_sparse_scene import (
    camera_id_from_line,
    ensure_colmap_text_model,
    image_name_from_line,
    load_eval_test_views,
    load_split,
)
from src.geometry.colmap import align_similarity_umeyama, qvec_to_rotmat
from src.utils.io import ensure_dir, save_csv, save_json


INSTANTSPLAT_ROOT = PROJECT_ROOT / "external_methods" / "InstantSplat"
DEFAULT_CKPT = (
    INSTANTSPLAT_ROOT
    / "mast3r"
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
DEFAULT_SCENES = ["tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"]
DEFAULT_SPLITS = [
    "5views_uniform",
    "5views_random",
    "5views_quality",
    "5views_quality_diverse",
]


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    qvec = np.empty(4, dtype=np.float64)
    trace = np.trace(rot)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qvec[0] = 0.25 * scale
        qvec[1] = (rot[2, 1] - rot[1, 2]) / scale
        qvec[2] = (rot[0, 2] - rot[2, 0]) / scale
        qvec[3] = (rot[1, 0] - rot[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qvec[0] = (rot[2, 1] - rot[1, 2]) / scale
            qvec[1] = 0.25 * scale
            qvec[2] = (rot[0, 1] + rot[1, 0]) / scale
            qvec[3] = (rot[0, 2] + rot[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qvec[0] = (rot[0, 2] - rot[2, 0]) / scale
            qvec[1] = (rot[0, 1] + rot[1, 0]) / scale
            qvec[2] = 0.25 * scale
            qvec[3] = (rot[1, 2] + rot[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qvec[0] = (rot[1, 0] - rot[0, 1]) / scale
            qvec[1] = (rot[0, 2] + rot[2, 0]) / scale
            qvec[2] = (rot[1, 2] + rot[2, 1]) / scale
            qvec[3] = 0.25 * scale
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec / np.linalg.norm(qvec)


def is_colmap_image_record(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) < 10:
        return False
    try:
        int(parts[0])
        [float(value) for value in parts[1:8]]
        int(parts[8])
    except ValueError:
        return False
    return True


def read_image_records(images_txt: Path) -> dict[str, dict]:
    records = {}
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        image_line = lines[idx].strip()
        if not image_line or image_line.startswith("#") or not is_colmap_image_record(image_line):
            idx += 1
            continue

        points_line = ""
        if idx + 1 < len(lines):
            candidate = lines[idx + 1].strip()
            if candidate and not candidate.startswith("#") and not is_colmap_image_record(candidate):
                points_line = candidate
                idx += 1

        parts = image_line.split()
        name = parts[9]
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        rot_w2c = qvec_to_rotmat(qvec)
        center = -rot_w2c.T @ tvec
        records[name] = {
            "image_id": int(parts[0]),
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": int(parts[8]),
            "name": name,
            "points_line": points_line,
            "rot_w2c": rot_w2c,
            "center": center,
        }
        idx += 1
    return records


def read_camera_lines(cameras_txt: Path) -> dict[int, str]:
    lines = cameras_txt.read_text(encoding="utf-8").splitlines()
    cameras = {}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        cameras[int(line.split()[0])] = line
    return cameras


def first_camera_params(cameras_txt: Path) -> list[str]:
    cameras = read_camera_lines(cameras_txt)
    if not cameras:
        raise ValueError(f"No cameras found in {cameras_txt}")
    first = cameras[sorted(cameras)[0]].split()
    return first[1:]


def parse_w2c_from_pose_array(pose_path: Path) -> list[np.ndarray]:
    poses = np.load(pose_path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected pose array shape: {poses.shape}")
    return [poses[i] for i in range(poses.shape[0])]


def camera_centers_from_records(records: dict[str, dict], names: list[str]) -> np.ndarray:
    return np.stack([records[name]["center"] for name in names], axis=0)


def write_test_sparse_model(
    source_dir: Path,
    scene: str,
    split: str,
    views: int,
    test_names: list[str],
    iteration: int,
) -> dict:
    text_model = ensure_colmap_text_model(scene)
    reference_records = read_image_records(text_model / "images.txt")
    train_names = load_split(scene, split)

    train_sparse = source_dir / f"sparse_{views}" / "0"
    inst_records = read_image_records(train_sparse / "images.txt")
    inst_names_by_id = [
        rec["name"] for rec in sorted(inst_records.values(), key=lambda rec: rec["image_id"])
    ]
    pose_path = (
        PROJECT_ROOT
        / "results"
        / "method_outputs"
        / scene
        / "instantsplat_eval"
        / split
        / "pose"
        / f"ours_{iteration}"
        / "pose_optimized.npy"
    )
    inst_poses_w2c = parse_w2c_from_pose_array(pose_path)
    inst_centers_by_name = {
        name: -pose[:3, :3].T @ pose[:3, 3]
        for name, pose in zip(inst_names_by_id, inst_poses_w2c)
    }

    common_train = [name for name in train_names if name in inst_centers_by_name]
    if len(common_train) < 3:
        raise ValueError("Need at least 3 train cameras to align InstantSplat and COLMAP coordinates")

    ref_train_centers = camera_centers_from_records(reference_records, common_train)
    inst_train_centers = np.stack([inst_centers_by_name[name] for name in common_train], axis=0)
    _aligned, transform = align_similarity_umeyama(ref_train_centers, inst_train_centers)
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)

    sparse_1 = ensure_dir(source_dir / f"sparse_{views}" / "1")
    camera_params = first_camera_params(train_sparse / "cameras.txt")

    camera_lines = ["# Camera list with one line of data per camera:"]
    image_lines = ["# Image list with two lines of data per image:"]

    for idx, name in enumerate(test_names, start=1):
        if name not in reference_records:
            raise ValueError(f"Test image not found in reference COLMAP model: {name}")
        ref = reference_records[name]
        ref_c2w_rot = ref["rot_w2c"].T
        ref_center = ref["center"]
        inst_center = scale * (rotation @ ref_center) + translation
        inst_c2w_rot = rotation @ ref_c2w_rot
        inst_w2c_rot = inst_c2w_rot.T
        inst_tvec = -inst_w2c_rot @ inst_center
        qvec = rotmat_to_qvec(inst_w2c_rot)

        camera_lines.append(" ".join([str(idx), *camera_params]))
        image_lines.append(
            " ".join(
                [
                    str(idx),
                    *(f"{value:.17g}" for value in qvec),
                    *(f"{value:.17g}" for value in inst_tvec),
                    str(idx),
                    name,
                ]
            )
        )
        image_lines.append("")

    (sparse_1 / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_1 / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    return {
        "num_test_images": len(test_names),
        "alignment_scale": scale,
        "alignment_translation": translation.tolist(),
    }


def copy_images(scene: str, names: list[str], image_dir: Path) -> None:
    raw_image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
    for name in names:
        src = raw_image_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        shutil.copy2(src, image_dir / name)


def materialize_train_source(scene: str, split: str, eval_test_views_path: str | None, clean: bool) -> tuple[Path, list[str], list[str]]:
    train_names = load_split(scene, split)
    test_names = load_eval_test_views(scene, eval_test_views_path)
    source_dir = PROJECT_ROOT / "datasets" / "instantsplat_eval" / scene / split
    if clean and source_dir.exists():
        shutil.rmtree(source_dir)
    image_dir = ensure_dir(source_dir / "images")

    copy_images(scene, train_names, image_dir)

    save_json(
        {
            "scene": scene,
            "split": split,
            "train_images": train_names,
            "test_images": test_names,
        },
        source_dir / "materialize_report.json",
    )
    return source_dir, train_names, test_names


def patch_torch_load_for_trusted_checkpoint():
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    return original_load


def run_init_geo(args, source_dir: Path, output_dir: Path) -> float:
    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(f"InstantSplat checkpoint not found: {DEFAULT_CKPT}")
    sys.path.insert(0, str(INSTANTSPLAT_ROOT))
    import init_geo

    init_geo.args = SimpleNamespace(device=args.device, focal_avg=False)
    original_load = patch_torch_load_for_trusted_checkpoint()
    start = time.perf_counter()
    try:
        with (output_dir / "init_geo_console.log").open("w", encoding="utf-8", errors="replace") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                init_geo.main(
                    source_path=str(source_dir),
                    model_path=str(output_dir),
                    ckpt_path=str(DEFAULT_CKPT),
                    device=args.device,
                    batch_size=1,
                    image_size=args.image_size,
                    schedule="cosine",
                    lr=0.01,
                    niter=args.geo_niter,
                    min_conf_thr=5,
                    llffhold=8,
                    n_views=args.views,
                    co_vis_dsp=True,
                    depth_thre=0.01,
                    conf_aware_ranking=False,
                    focal_avg=False,
                    infer_video=True,
                )
    finally:
        torch.load = original_load
    return time.perf_counter() - start


def run_subprocess(command: list[str], cwd: Path, log_path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode, time.perf_counter() - start


def run_train(args, source_dir: Path, output_dir: Path) -> tuple[int, float]:
    command = [
        sys.executable,
        "train.py",
        "--source_path",
        str(source_dir),
        "--model_path",
        str(output_dir),
        "--n_views",
        str(args.views),
        "--iterations",
        str(args.train_iterations),
        "--resolution",
        str(args.resolution),
        "--test_iterations",
        str(args.train_iterations),
        "--save_iterations",
        str(args.train_iterations),
        "--disable_viewer",
    ]
    return run_subprocess(command, INSTANTSPLAT_ROOT, output_dir / "train_console.log")


def run_render(args, source_dir: Path, output_dir: Path) -> tuple[int, float]:
    command = [
        sys.executable,
        "render.py",
        "--source_path",
        str(source_dir),
        "--model_path",
        str(output_dir),
        "--iterations",
        str(args.train_iterations),
        "--n_views",
        str(args.views),
        "--resolution",
        str(args.resolution),
        "--skip_train",
        "--eval",
        "--optim_test_pose_iter",
        str(args.optim_test_pose_iter),
    ]
    return run_subprocess(command, INSTANTSPLAT_ROOT, output_dir / "render_console.log")


def run_one(args, scene: str, split: str) -> dict:
    output_dir = ensure_dir(PROJECT_ROOT / "results" / "method_outputs" / scene / "instantsplat_eval" / split)
    row = {
        "scene": scene,
        "method": "InstantSplat",
        "split": split,
        "views": args.views,
        "train_iterations": args.train_iterations,
        "status": "started",
        "output_dir": str(output_dir),
        "error": "",
    }
    try:
        eval_test_views_path = f"datasets/processed/{scene}/eval_test_views.json"
        source_dir, _train_names, test_names = materialize_train_source(
            scene, split, eval_test_views_path, clean=not args.keep_existing_source
        )
        row["source_dir"] = str(source_dir)
        row["num_test_images"] = len(test_names)
        row["init_geo_sec"] = run_init_geo(args, source_dir, output_dir)
        train_returncode, train_sec = run_train(args, source_dir, output_dir)
        row["train_returncode"] = train_returncode
        row["train_sec"] = train_sec
        if train_returncode != 0:
            row["status"] = "train_failed"
            return row

        copy_images(scene, test_names, source_dir / "images")
        row.update(write_test_sparse_model(source_dir, scene, split, args.views, test_names, args.train_iterations))
        render_returncode, render_sec = run_render(args, source_dir, output_dir)
        row["render_returncode"] = render_returncode
        row["render_sec"] = render_sec
        if render_returncode != 0:
            row["status"] = "render_failed"
            return row

        metrics = evaluate(output_dir, f"ours_{args.train_iterations}")
        row["psnr"] = metrics["psnr"]
        row["ssim"] = metrics["ssim"]
        row["lpips"] = metrics["lpips"]
        row["status"] = "success"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        save_json(row, output_dir / "eval_run_report.json")
    return row


def append_rows(rows: list[dict], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--geo-niter", type=int, default=300)
    parser.add_argument("--train-iterations", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--optim-test-pose-iter", type=int, default=0)
    parser.add_argument("--keep-existing-source", action="store_true")
    parser.add_argument("--csv", default="results/csv/foundation/instantsplat_eval_5views.csv")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    rows = []
    for scene in args.scenes:
        for split in args.splits:
            row = run_one(args, scene, split)
            rows.append(row)
            append_rows(rows, csv_path)
            print(json.dumps(row, indent=2))

    print(f"Saved InstantSplat eval benchmark to {csv_path}")


if __name__ == "__main__":
    main()
