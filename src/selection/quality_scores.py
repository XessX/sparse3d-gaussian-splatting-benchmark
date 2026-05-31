from pathlib import Path
import cv2
import numpy as np
from skimage.measure import shannon_entropy

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def exposure_score(gray: np.ndarray) -> float:
    # Good exposure is near middle intensity and avoids clipping.
    mean = gray.mean() / 255.0
    clipped_dark = np.mean(gray < 5)
    clipped_bright = np.mean(gray > 250)
    center_score = 1.0 - abs(mean - 0.5) * 2.0
    clipping_penalty = clipped_dark + clipped_bright
    return float(max(0.0, center_score - clipping_penalty))

def entropy_score(gray: np.ndarray) -> float:
    return float(shannon_entropy(gray))

def feature_count_score(gray: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints = orb.detect(gray, None)
    return float(len(keypoints))

def normalize_scores(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return []
    min_v, max_v = float(arr.min()), float(arr.max())
    if max_v - min_v < 1e-9:
        return [0.5 for _ in arr]
    return ((arr - min_v) / (max_v - min_v)).tolist()

def rank_images_by_quality(image_dir: str | Path, weights: dict | None = None) -> list[dict]:
    image_dir = Path(image_dir)
    weights = weights or {
        "sharpness": 0.30,
        "exposure": 0.25,
        "entropy": 0.20,
        "feature_count": 0.25,
    }

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    raw = []
    for path in image_paths:
        gray = load_gray(path)
        raw.append({
            "filename": path.name,
            "sharpness": sharpness_score(gray),
            "exposure": exposure_score(gray),
            "entropy": entropy_score(gray),
            "feature_count": feature_count_score(gray),
        })

    for key in ["sharpness", "exposure", "entropy", "feature_count"]:
        normalized = normalize_scores([r[key] for r in raw])
        for r, v in zip(raw, normalized):
            r[f"{key}_norm"] = v

    for r in raw:
        r["quality_score"] = sum(
            weights.get(key, 0.0) * r[f"{key}_norm"]
            for key in ["sharpness", "exposure", "entropy", "feature_count"]
        )

    return sorted(raw, key=lambda x: x["quality_score"], reverse=True)

def select_top_quality_views(image_dir: str | Path, n: int, weights: dict | None = None) -> list[str]:
    ranked = rank_images_by_quality(image_dir, weights)
    if n > len(ranked):
        raise ValueError(f"Requested {n} views but only {len(ranked)} images exist")
    return [r["filename"] for r in ranked[:n]]
