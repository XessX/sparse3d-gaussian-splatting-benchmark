import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.executables import find_executable
from src.utils.io import ensure_dir, save_json


def scene_name(scene: str) -> str:
    return Path(scene).name


def load_split(scene: str, split_name: str) -> list[str]:
    split_path = (
        PROJECT_ROOT
        / "datasets"
        / "processed"
        / scene
        / "splits"
        / f"{scene_name(scene)}_splits.json"
    )
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    try:
        return data["splits"][split_name]
    except KeyError as exc:
        available = ", ".join(sorted(data.get("splits", {}).keys()))
        raise KeyError(f"Split '{split_name}' not found. Available splits: {available}") from exc


def ensure_colmap_text_model(scene: str) -> Path:
    output_dir = ensure_dir(PROJECT_ROOT / "datasets" / "processed" / scene / "colmap_text")
    expected = output_dir / "images.txt"
    if expected.exists():
        return output_dir

    source_sparse = PROJECT_ROOT / "datasets" / "raw" / scene / "sparse" / "0"
    if not source_sparse.exists():
        raise FileNotFoundError(f"COLMAP sparse model not found: {source_sparse}")

    colmap = find_executable("colmap")
    if not colmap:
        raise FileNotFoundError("COLMAP executable not found.")

    subprocess.run(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(source_sparse),
            "--output_path",
            str(output_dir),
            "--output_type",
            "TXT",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_dir


def split_colmap_images(lines: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    header = [line for line in lines if line.startswith("#")]
    payload = [line.rstrip("\n") for line in lines if line.strip() and not line.startswith("#")]
    if len(payload) % 2 != 0:
        raise ValueError("COLMAP images.txt payload should contain pairs of image and point lines.")
    pairs = [(payload[i], payload[i + 1]) for i in range(0, len(payload), 2)]
    return header, pairs


def image_name_from_line(line: str) -> str:
    parts = line.split()
    if len(parts) < 10:
        raise ValueError(f"Malformed image line: {line[:120]}")
    return parts[9]


def image_id_from_line(line: str) -> int:
    return int(line.split()[0])


def camera_id_from_line(line: str) -> int:
    return int(line.split()[8])


def point_ids_from_points2d_line(line: str) -> set[int]:
    parts = line.split()
    point_ids = set()
    for idx in range(2, len(parts), 3):
        pid = int(float(parts[idx]))
        if pid >= 0:
            point_ids.add(pid)
    return point_ids


def filter_points_line(line: str, selected_image_ids: set[int]) -> str | None:
    parts = line.split()
    if len(parts) < 8:
        return None
    prefix = parts[:8]
    track = parts[8:]
    if len(track) % 2 != 0:
        return None
    filtered_pairs: list[str] = []
    for idx in range(0, len(track), 2):
        image_id = int(track[idx])
        point2d_idx = track[idx + 1]
        if image_id in selected_image_ids:
            filtered_pairs.extend([str(image_id), point2d_idx])
    if not filtered_pairs:
        return None
    return " ".join(prefix + filtered_pairs)


def load_eval_test_views(scene: str, path: str | None = None) -> list[str]:
    if path is None:
        return []
    test_path = PROJECT_ROOT / path
    if not test_path.exists():
        return []
    data = json.loads(test_path.read_text(encoding="utf-8"))
    return data.get("test_images", [])


def materialize_sparse_scene(
    scene: str,
    split_name: str,
    output_root: str,
    eval_test_views_path: str | None = None,
) -> dict:
    train_names = set(load_split(scene, split_name))
    test_names = set(load_eval_test_views(scene, eval_test_views_path))
    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"Train/test overlap for {split_name}: {sorted(overlap)}")
    selected_names = train_names | test_names
    raw_scene_dir = PROJECT_ROOT / "datasets" / "raw" / scene
    raw_image_dir = raw_scene_dir / "images"
    if not raw_image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {raw_image_dir}")

    text_model = ensure_colmap_text_model(scene)
    output_dir = PROJECT_ROOT / output_root / scene / split_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_image_dir = ensure_dir(output_dir / "images")
    output_sparse_dir = ensure_dir(output_dir / "sparse" / "0")

    for name in selected_names:
        src = raw_image_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Selected image missing: {src}")
        shutil.copy2(src, output_image_dir / name)

    cameras_lines = (text_model / "cameras.txt").read_text(encoding="utf-8").splitlines()
    image_lines = (text_model / "images.txt").read_text(encoding="utf-8").splitlines()
    points_lines = (text_model / "points3D.txt").read_text(encoding="utf-8").splitlines()

    image_header, image_pairs = split_colmap_images(image_lines)
    selected_pairs = [
        pair for pair in image_pairs if image_name_from_line(pair[0]) in selected_names
    ]
    if len(selected_pairs) != len(selected_names):
        found = {image_name_from_line(pair[0]) for pair in selected_pairs}
        missing = sorted(selected_names - found)
        raise ValueError(f"Selected images not found in COLMAP model: {missing}")

    train_pairs = [
        pair for pair in selected_pairs if image_name_from_line(pair[0]) in train_names
    ]
    test_pairs = [
        pair for pair in selected_pairs if image_name_from_line(pair[0]) in test_names
    ]
    selected_image_ids = {image_id_from_line(pair[0]) for pair in train_pairs}
    selected_camera_ids = {camera_id_from_line(pair[0]) for pair in selected_pairs}
    selected_point_ids: set[int] = set()
    for _, points2d_line in train_pairs:
        selected_point_ids.update(point_ids_from_points2d_line(points2d_line))

    camera_output_lines = [
        line
        for line in cameras_lines
        if line.startswith("#") or (line.strip() and int(line.split()[0]) in selected_camera_ids)
    ]
    (output_sparse_dir / "cameras.txt").write_text(
        "\n".join(camera_output_lines) + "\n", encoding="utf-8"
    )

    image_output_lines = image_header.copy()
    for image_line, points2d_line in selected_pairs:
        image_output_lines.extend([image_line, points2d_line])
    (output_sparse_dir / "images.txt").write_text(
        "\n".join(image_output_lines) + "\n", encoding="utf-8"
    )
    if test_names:
        (output_sparse_dir / "test.txt").write_text(
            "\n".join(sorted(test_names)) + "\n", encoding="utf-8"
        )

    point_output_lines = [line for line in points_lines if line.startswith("#")]
    for line in points_lines:
        if not line.strip() or line.startswith("#"):
            continue
        point_id = int(line.split()[0])
        if point_id not in selected_point_ids:
            continue
        filtered = filter_points_line(line, selected_image_ids)
        if filtered:
            point_output_lines.append(filtered)
    (output_sparse_dir / "points3D.txt").write_text(
        "\n".join(point_output_lines) + "\n", encoding="utf-8"
    )

    report = {
        "scene": scene,
        "split_name": split_name,
        "output_dir": str(output_dir),
        "num_images": len(selected_pairs),
        "num_train_images": len(train_pairs),
        "num_test_images": len(test_pairs),
        "num_cameras": len(selected_camera_ids),
        "num_observed_points": len(point_output_lines) - len(
            [line for line in point_output_lines if line.startswith("#")]
        ),
        "train_images": sorted(train_names),
        "test_images": sorted(test_names),
    }
    save_json(report, output_dir / "materialize_report.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--eval-test-views",
        default=None,
        help="Optional JSON file containing a test_images list.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/sparse_scenes",
        help="Root where runnable sparse-view scene folders are written.",
    )
    args = parser.parse_args()

    report = materialize_sparse_scene(
        args.scene, args.split, args.output_root, args.eval_test_views
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
