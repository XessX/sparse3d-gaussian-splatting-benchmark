import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=255))

def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 3:
        return float(structural_similarity(target, pred, channel_axis=2, data_range=255))
    return float(structural_similarity(target, pred, data_range=255))

def safe_metric(metric_fn, pred, target):
    try:
        return metric_fn(pred, target)
    except Exception:
        return None
