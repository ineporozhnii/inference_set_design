import numpy as np
from scipy.stats import entropy


def get_classification_uncertainty(class_probs: np.ndarray, n_classes: int) -> np.ndarray:
    batch_dim, class_dim = class_probs.shape
    assert class_dim == n_classes
    max_class_prob = np.max(class_probs, axis=1)
    u = (n_classes / (n_classes - 1)) * (1 - max_class_prob)
    return u


def get_ensemble_disagreement(ensmbl_class_probs: np.ndarray, n_classes: int) -> np.ndarray:
    batch_dim, ensemble_dim, class_dim = ensmbl_class_probs.shape
    assert class_dim == n_classes
    ensmbl_class_preds = np.argmax(ensmbl_class_probs, axis=2)
    ensmbl_pred_counts = (
        np.stack([np.sum(ensmbl_class_preds == c, axis=1) for c in range(n_classes)], axis=-1) / ensemble_dim
    )
    class_pred_ensmbl_H = entropy(ensmbl_pred_counts, axis=1)
    return class_pred_ensmbl_H


def get_ensemble_std(ensmbl_preds: np.ndarray) -> np.ndarray:
    ensmbl_std = np.std(ensmbl_preds, axis=-1)
    return ensmbl_std
