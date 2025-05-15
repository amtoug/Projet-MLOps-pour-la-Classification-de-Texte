from scipy.stats import ks_2samp
import sys
import numpy as np


def detect_drift(ref: np.ndarray, new: np.ndarray, threshold: float = 0.01) -> bool:
    if ref.size == 0 or new.shape[1] != ref.shape[1]:
        return False  # Trop tôt ou incompatible
    for i in range(ref.shape[1]):
        _, p = ks_2samp(ref[:, i], new[:, i])
        if p < threshold:
            return True
    return False