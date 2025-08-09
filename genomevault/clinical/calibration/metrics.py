from __future__ import annotations

"""Metrics module."""
import numpy as np


def _safe_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-12, 1.0 - 1e-12)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via Mann–Whitney U statistic (no sklearn)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Rank all scores (average rank for ties)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    # tie adjustment: average ranks in equal score groups
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            mean_rank = np.mean(ranks[order][i : j + 1])
            ranks[order][i : j + 1] = mean_rank
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return precision and recall arrays sorted by descending score."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)  # descending
    y_true = y_true[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y_true == 1).sum()), 1)
    # prepend (0,1) for conventional curve
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return precision, recall


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Sklearn-like AP (area under precision-recall curve using step interpolation)."""
    p, r = precision_recall_curve(y_true, y_score)
    # integrate precision over recall using step-wise method
    ap = 0.0
    for i in range(1, len(p)):
        dr = r[i] - r[i - 1]
        ap += p[i] * dr
    return float(ap)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score.

        Args:
            y_true: Y true.
            y_prob: Y prob.

        Returns:
            Float result.
        """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = _safe_prob(np.asarray(y_prob, dtype=np.float64))
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, mean_pred, frac_positives) using equal-width bins in [0,1]."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = _safe_prob(np.asarray(y_prob, dtype=np.float64))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        mask = idx == k
        if mask.sum() > 0:
            mean_pred[k] = y_prob[mask].mean()
            frac_pos[k] = y_true[mask].mean()
            counts[k] = mask.sum()
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return bin_centers, mean_pred, frac_pos


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Ece.

        Args:
            y_true: Y true.
            y_prob: Y prob.
            n_bins: N bins.

        Returns:
            Float result.
        """
    bins, mean_pred, frac_pos = calibration_curve(y_true, y_prob, n_bins=n_bins)
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = _safe_prob(np.asarray(y_prob, dtype=np.float64))
    # recompute counts for weight
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bin_edges, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    counts = np.array([(idx == k).sum() for k in range(n_bins)], dtype=np.float64)
    weights = counts / max(1.0, counts.sum())
    return float(np.sum(weights * np.abs(frac_pos - mean_pred)))


def mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Mce.

        Args:
            y_true: Y true.
            y_prob: Y prob.
            n_bins: N bins.

        Returns:
            Float result.
        """
    _, mean_pred, frac_pos = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(np.max(np.abs(frac_pos - mean_pred)))


def confusion_at(
    """Confusion at.

        Args:
            y_true: Y true.
            y_prob: Y prob.
            threshold: Threshold value.

        Returns:
            Operation result.
        """
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_hat = (y_prob >= threshold).astype(np.int32)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    rec = sens
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def youdens_j_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Compute threshold that maximizes Youden's J = sensitivity + specificity - 1."""
    # candidate thresholds are unique probabilities
    probs = np.unique(np.asarray(y_prob, dtype=np.float64))
    best_t = 0.5
    best_j = -1.0
    best_stats = {}
    for t in probs:
        stats = confusion_at(y_true, y_prob, float(t))
        j = stats["sensitivity"] + stats["specificity"] - 1.0
        if j > best_j:
            best_t = float(t)
            best_j = j
            best_stats = stats
    return best_t, best_stats
