from __future__ import annotations

from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP qvec order (qw, qx, qy, qz) to a rotation matrix."""
    q0, q1, q2, q3 = qvec
    return np.array(
        [
            [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q3 * q1 + 2 * q0 * q2],
            [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
            [2 * q3 * q1 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2],
        ],
        dtype=np.float64,
    )


def split_colmap_images(lines: list[str]) -> list[tuple[str, str]]:
    payload = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(payload) % 2 != 0:
        raise ValueError("COLMAP images.txt payload should contain image/points line pairs.")
    return [(payload[i], payload[i + 1]) for i in range(0, len(payload), 2)]


def read_colmap_camera_centers(images_txt: str | Path) -> dict[str, np.ndarray]:
    images_txt = Path(images_txt)
    if not images_txt.exists():
        raise FileNotFoundError(f"COLMAP images.txt not found: {images_txt}")

    centers: dict[str, np.ndarray] = {}
    for image_line, _points_line in split_colmap_images(images_txt.read_text(encoding="utf-8").splitlines()):
        parts = image_line.split()
        if len(parts) < 10:
            raise ValueError(f"Malformed COLMAP image line: {image_line[:120]}")
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        name = parts[9]
        rot = qvec_to_rotmat(qvec)
        centers[name] = -rot.T @ tvec
    return centers


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Return camera centers from OpenCV world-to-camera extrinsics [..., 3, 4]."""
    rot = extrinsics[..., :3, :3]
    trans = extrinsics[..., :3, 3]
    return -np.einsum("...ji,...j->...i", rot, trans)


def align_similarity_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Align source points to target with a similarity transform."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected Nx3 arrays with equal shape, got {source.shape} and {target.shape}")
    if len(source) == 0:
        raise ValueError("Cannot align empty point sets")

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    src_var = float(np.mean(np.sum(src_centered**2, axis=1)))

    if src_var < 1e-12:
        aligned = np.repeat(tgt_mean[None, :], len(source), axis=0)
        return aligned, {"scale": 0.0, "rotation": np.eye(3).tolist(), "translation": tgt_mean.tolist()}

    cov = (tgt_centered.T @ src_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    diag = np.diag([1.0, 1.0, sign])
    rotation = u @ diag @ vt
    scale = float(np.trace(np.diag(singular_values) @ diag) / src_var)
    translation = tgt_mean - scale * rotation @ src_mean
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def trajectory_extent(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return 0.0
    centered = points - points.mean(axis=0)
    return float(2.0 * np.max(np.linalg.norm(centered, axis=1)))
