import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from materialize_sparse_scene import ensure_colmap_text_model, image_name_from_line, split_colmap_images
from src.selection.quality_scores import rank_images_by_quality


def scene_leaf(scene: str) -> str:
    return Path(scene).name


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def load_camera_centers(scene: str) -> dict[str, np.ndarray]:
    text_model = ensure_colmap_text_model(scene)
    lines = (text_model / "images.txt").read_text(encoding="utf-8").splitlines()
    _, pairs = split_colmap_images(lines)
    centers: dict[str, np.ndarray] = {}
    for image_line, _ in pairs:
        parts = image_line.split()
        name = image_name_from_line(image_line)
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        centers[name] = -qvec_to_rotmat(qvec).T @ tvec
    return centers


def normalized_quality(scene: str) -> dict[str, float]:
    image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
    ranked = rank_images_by_quality(image_dir)
    return {row["filename"]: float(row["quality_score"]) for row in ranked}


def select_quality_diverse(
    quality: dict[str, float],
    centers: dict[str, np.ndarray],
    n: int,
    quality_weight: float,
    diversity_weight: float,
) -> list[str]:
    candidates = sorted(set(quality) & set(centers))
    if n > len(candidates):
        raise ValueError(f"Requested {n} views but only {len(candidates)} candidate images exist")

    center_stack = np.stack([centers[name] for name in candidates], axis=0)
    max_dist = float(np.linalg.norm(center_stack[:, None, :] - center_stack[None, :, :], axis=-1).max())
    max_dist = max(max_dist, 1e-9)

    selected = [max(candidates, key=lambda name: quality[name])]
    remaining = set(candidates) - set(selected)
    while len(selected) < n:
        best_name = None
        best_score = -1.0
        for name in remaining:
            diversity = min(
                float(np.linalg.norm(centers[name] - centers[selected_name])) / max_dist
                for selected_name in selected
            )
            score = quality_weight * quality[name] + diversity_weight * diversity
            if score > best_score:
                best_name = name
                best_score = score
        selected.append(best_name)
        remaining.remove(best_name)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="tandt/train")
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 5, 8, 12, 20])
    parser.add_argument("--quality-weight", type=float, default=0.45)
    parser.add_argument("--diversity-weight", type=float, default=0.55)
    args = parser.parse_args()

    split_path = (
        PROJECT_ROOT
        / "datasets"
        / "processed"
        / args.scene
        / "splits"
        / f"{scene_leaf(args.scene)}_splits.json"
    )
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    quality = normalized_quality(args.scene)
    centers = load_camera_centers(args.scene)

    for n in args.views:
        data["splits"][f"{n}views_quality_diverse"] = select_quality_diverse(
            quality, centers, n, args.quality_weight, args.diversity_weight
        )

    data["quality_diverse_selection"] = {
        "quality_weight": args.quality_weight,
        "diversity_weight": args.diversity_weight,
        "notes": "Greedy selection using image quality and minimum camera-center distance to selected views.",
    }
    split_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved quality-diverse splits to {split_path}")
    for n in args.views:
        print(f"{n}views_quality_diverse: {data['splits'][f'{n}views_quality_diverse']}")


if __name__ == "__main__":
    main()
