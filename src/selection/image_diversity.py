from pathlib import Path

import cv2
import numpy as np

from src.selection.quality_scores import normalize_scores, rank_images_by_quality


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def read_image_paths(image_dir: str | Path, exclude: set[str] | None = None) -> list[Path]:
    image_dir = Path(image_dir)
    exclude = exclude or set()
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTS and path.name not in exclude
    )


def load_small_rgb(path: Path, max_side: int = 384) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def color_histogram(rgb: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def orb_descriptor(rgb: np.ndarray) -> tuple[int, np.ndarray | None]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return len(keypoints), descriptors


def histogram_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    correlation = cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL)
    return float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))


def orb_match_distance(desc_a: np.ndarray | None, desc_b: np.ndarray | None) -> float:
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 1.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)
    if not matches:
        return 1.0
    distances = np.array([match.distance for match in matches], dtype=np.float32)
    good_ratio = float(np.mean(distances < 48.0))
    median_distance = float(np.median(distances) / 256.0)
    return float(np.clip(0.65 * (1.0 - good_ratio) + 0.35 * median_distance, 0.0, 1.0))


def build_image_feature_table(
    image_dir: str | Path,
    exclude: set[str] | None = None,
    candidate_pool: int = 80,
) -> list[dict]:
    image_dir = Path(image_dir)
    exclude = exclude or set()
    ranked_quality = [
        row for row in rank_images_by_quality(image_dir) if row["filename"] not in exclude
    ]
    if not ranked_quality:
        raise ValueError(f"No candidate images found in {image_dir}")

    pool = ranked_quality[: min(candidate_pool, len(ranked_quality))]
    quality_values = normalize_scores([row["quality_score"] for row in pool])
    features = []
    for row, quality_norm in zip(pool, quality_values):
        path = image_dir / row["filename"]
        rgb = load_small_rgb(path)
        feature_count, descriptor = orb_descriptor(rgb)
        features.append(
            {
                **row,
                "quality_score_norm": quality_norm,
                "feature_count_image_diverse": feature_count,
                "histogram": color_histogram(rgb),
                "descriptor": descriptor,
            }
        )
    return features


def pairwise_image_distance(row_a: dict, row_b: dict, hist_weight: float = 0.45, orb_weight: float = 0.55) -> float:
    hist_dist = histogram_distance(row_a["histogram"], row_b["histogram"])
    orb_dist = orb_match_distance(row_a["descriptor"], row_b["descriptor"])
    return float(np.clip(hist_weight * hist_dist + orb_weight * orb_dist, 0.0, 1.0))


def select_quality_image_diverse(
    image_dir: str | Path,
    n: int,
    exclude: set[str] | None = None,
    candidate_pool: int = 80,
    quality_weight: float = 0.45,
    diversity_weight: float = 0.55,
) -> tuple[list[str], list[dict]]:
    features = build_image_feature_table(image_dir, exclude=exclude, candidate_pool=max(candidate_pool, n))
    if n > len(features):
        raise ValueError(f"Requested {n} views but only {len(features)} candidate images exist.")

    selected_indices = [0]
    remaining = set(range(1, len(features)))
    selection_rows = [
        {
            "rank": 1,
            "filename": features[0]["filename"],
            "quality_score": features[0]["quality_score"],
            "quality_score_norm": features[0]["quality_score_norm"],
            "diversity_score": 0.0,
            "combined_score": features[0]["quality_score_norm"],
        }
    ]

    while len(selected_indices) < n:
        best_idx = None
        best_row = None
        for idx in remaining:
            diversity = min(
                pairwise_image_distance(features[idx], features[selected_idx])
                for selected_idx in selected_indices
            )
            combined = quality_weight * features[idx]["quality_score_norm"] + diversity_weight * diversity
            if best_row is None or combined > best_row["combined_score"]:
                best_idx = idx
                best_row = {
                    "rank": len(selected_indices) + 1,
                    "filename": features[idx]["filename"],
                    "quality_score": features[idx]["quality_score"],
                    "quality_score_norm": features[idx]["quality_score_norm"],
                    "diversity_score": diversity,
                    "combined_score": combined,
                }
        if best_idx is None or best_row is None:
            raise RuntimeError("Image-diverse selection failed to choose a candidate.")
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        selection_rows.append(best_row)

    return sorted(row["filename"] for row in selection_rows), selection_rows
