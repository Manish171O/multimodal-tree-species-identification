"""Classification metrics in NumPy (no sklearn dependency). Extended in Phase 5."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """(C, C) counts, rows = true class, cols = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall, F1 and support from a confusion matrix (0 where undefined)."""
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(0).astype(np.float64)
    sup = cm.sum(1).astype(np.float64)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, sup


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    """Accuracy, macro-F1 (over classes present in y_true), weighted-F1, macro precision/recall."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    prec, rec, f1, sup = precision_recall_f1(cm)
    present = sup > 0
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * sup).sum() / max(sup.sum(), 1)),
        "macro_precision": float(prec[present].mean()) if present.any() else 0.0,
        "macro_recall": float(rec[present].mean()) if present.any() else 0.0,
    }


def expected_calibration_error(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15
                               ) -> tuple[float, dict[str, np.ndarray]]:
    """Top-label ECE with equal-width confidence bins (Guo et al., 2017).

    prob: (N, C) predicted probabilities. Returns (ece, bins) where bins has per-bin
    'confidence', 'accuracy', 'count' (N_bins,) and 'edges' (N_bins + 1,).
    """
    conf = prob.max(1)                                                 # (N,)
    correct = (prob.argmax(1) == y_true).astype(np.float64)            # (N,)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)  # (N,) bin id, (lo, hi]
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    s_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    s_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_conf = np.where(cnt > 0, s_conf / cnt, 0.0)
        b_acc = np.where(cnt > 0, s_acc / cnt, 0.0)
    ece = float((cnt / max(len(y_true), 1) * np.abs(b_acc - b_conf)).sum())
    return ece, {"confidence": b_conf, "accuracy": b_acc, "count": cnt, "edges": edges}


def top_k_accuracy(prob: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    """Fraction of samples whose true class is among the k most probable."""
    topk = np.argpartition(-prob, kth=min(k, prob.shape[1]) - 1, axis=1)[:, :k]   # (N, k)
    return float((topk == y_true[:, None]).any(1).mean()) if len(y_true) else 0.0


def full_metrics(y_true: np.ndarray, prob: np.ndarray, num_classes: int, n_bins: int = 15) -> dict[str, float]:
    """basic_metrics + top-5 accuracy + ECE + NLL for one evaluation subset."""
    pred = prob.argmax(1)
    res = basic_metrics(y_true, pred, num_classes)
    res["top5_accuracy"] = top_k_accuracy(prob, y_true, 5)
    res["ece"] = expected_calibration_error(prob, y_true, n_bins)[0]
    res["nll"] = float(-np.log(np.clip(prob[np.arange(len(y_true)), y_true], 1e-12, None)).mean()) if len(y_true) else 0.0
    res["n"] = int(len(y_true))
    return res


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, names: list[str] | None = None,
                     train_counts: np.ndarray | None = None) -> list[dict]:
    """One row per class that appears in y_true or y_pred: precision, recall, F1, support, train count."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    prec, rec, f1, sup = precision_recall_f1(cm)
    rows = []
    for c in range(num_classes):
        if sup[c] == 0 and cm[:, c].sum() == 0:
            continue
        rows.append({"class_id": c, "species": names[c] if names else str(c), "precision": float(prec[c]),
                     "recall": float(rec[c]), "f1": float(f1[c]), "support": int(sup[c]),
                     "train_count": int(train_counts[c]) if train_counts is not None else -1})
    return rows
