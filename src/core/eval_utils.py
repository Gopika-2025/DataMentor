# Shared metrics and plotting utils 
# src/core/eval_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import label_binarize


# ---------------------------
# General helpers
# ---------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None

def _to_list(a: np.ndarray) -> List[Any]:
    return np.asarray(a).tolist()

def infer_supervised_task(y: Union[np.ndarray, List, Tuple]) -> str:
    """
    Heuristic:
    - numeric with many unique values -> regression
    - otherwise classification
    """
    y = np.asarray(y)
    if y.dtype.kind in {"i", "u", "f"}:
        # consider regression if many distinct continuous-ish values
        nun = len(np.unique(y[~np.isnan(y)])) if y.dtype.kind in {"f"} else len(np.unique(y))
        if nun > 15:
            return "regression"
    return "classification"


# ---------------------------
# Classification metrics
# ---------------------------

@dataclass
class ClassifInputs:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None  # shape (n_samples,) for binary; (n_samples, n_classes) for multiclass
    labels: Optional[List[Any]] = None    # explicit label order (for cm/report)
    average: str = "weighted"             # "micro" | "macro" | "weighted"
    pos_label: Optional[Any] = None       # for binary specificity etc.

def _ensure_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[Any]]) -> List[Any]:
    if labels is not None:
        return list(labels)
    return sorted(list(set(np.unique(y_true)).union(set(np.unique(y_pred)))))

def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    lbs = _ensure_labels(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=lbs)
    return {
        "labels": [str(x) for x in lbs],
        "matrix": _to_list(cm.astype(int)),
    }

def _binary_specificity(y_true: np.ndarray, y_pred: np.ndarray, pos_label: Any) -> Optional[float]:
    try:
        # Map to {0,1} with pos_label as 1
        y_true_bin = (y_true == pos_label).astype(int)
        y_pred_bin = (y_pred == pos_label).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        return _safe_float(spec)
    except Exception:
        return None

def classification_metrics(ci: ClassifInputs) -> Dict[str, Any]:
    y_true = np.asarray(ci.y_true)
    y_pred = np.asarray(ci.y_pred)
    y_proba = None if ci.y_proba is None else np.asarray(ci.y_proba)
    task = "classification"
    lbs = _ensure_labels(y_true, y_pred, ci.labels)
    res: Dict[str, Any] = {"task": task}

    # Basic scores
    res["accuracy"] = _safe_float(accuracy_score(y_true, y_pred))
    res["precision_" + ci.average] = _safe_float(precision_score(y_true, y_pred, average=ci.average, zero_division=0))
    res["recall_" + ci.average] = _safe_float(recall_score(y_true, y_pred, average=ci.average, zero_division=0))
    res["f1_" + ci.average] = _safe_float(f1_score(y_true, y_pred, average=ci.average, zero_division=0))
    # MCC (balanced, robust)
    try:
        res["mcc"] = _safe_float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        res["mcc"] = None

    # Confusion matrix
    res["confusion_matrix"] = confusion_matrix_dict(y_true, y_pred, lbs)

    # Specificity (binary only, requires pos_label)
    if ci.pos_label is not None and len(lbs) == 2:
        res["specificity"] = _binary_specificity(y_true, y_pred, ci.pos_label)

    # Probabilistic metrics + ROC/PR curves if proba provided
    if y_proba is not None:
        try:
            # Shape handling
            n_classes = len(lbs)
            if n_classes == 2:
                # Accept (n_samples,) or (n_samples,2). Use prob of positive (pos_label if given else lbs[1])
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    # choose the column that corresponds to positive class
                    pos = ci.pos_label if ci.pos_label is not None else lbs[1]
                    pos_idx = lbs.index(pos)
                    p_pos = y_proba[:, pos_idx]
                else:
                    p_pos = y_proba.reshape(-1)
                # Log loss
                try:
                    res["log_loss"] = _safe_float(log_loss(y_true, np.c_[1 - p_pos, p_pos], labels=lbs))
                except Exception:
                    res["log_loss"] = None
                # ROC
                try:
                    y_bin = (y_true == (ci.pos_label if ci.pos_label is not None else lbs[1])).astype(int)
                    fpr, tpr, thr = roc_curve(y_bin, p_pos)
                    res["roc_curve"] = {"fpr": _to_list(fpr), "tpr": _to_list(tpr), "thresholds": _to_list(thr), "auc": _safe_float(auc(fpr, tpr))}
                except Exception:
                    res["roc_curve"] = None
                # PR
                try:
                    prec, rec, thr = precision_recall_curve(y_bin, p_pos)
                    ap = average_precision_score(y_bin, p_pos)
                    res["pr_curve"] = {"precision": _to_list(prec), "recall": _to_list(rec), "thresholds": _to_list(thr), "ap": _safe_float(ap)}
                except Exception:
                    res["pr_curve"] = None
            else:
                # Multiclass one-vs-rest curves + macro AUC/AP
                try:
                    y_bin = label_binarize(y_true, classes=lbs)  # shape (n, K)
                except Exception:
                    y_bin = None
                macro_aucs = []
                macro_aps = []
                roc_curves = {}
                pr_curves = {}
                # y_proba expected shape (n, K)
                if y_bin is not None and y_proba.ndim == 2 and y_proba.shape[1] == len(lbs):
                    for i, lab in enumerate(lbs):
                        try:
                            fpr, tpr, thr = roc_curve(y_bin[:, i], y_proba[:, i])
                            roc_curves[str(lab)] = {
                                "fpr": _to_list(fpr), "tpr": _to_list(tpr),
                                "thresholds": _to_list(thr), "auc": _safe_float(auc(fpr, tpr))
                            }
                            macro_aucs.append(auc(fpr, tpr))
                        except Exception:
                            pass
                        try:
                            prec, rec, thr = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                            pr_curves[str(lab)] = {
                                "precision": _to_list(prec), "recall": _to_list(rec),
                                "thresholds": _to_list(thr), "ap": _safe_float(average_precision_score(y_bin[:, i], y_proba[:, i]))
                            }
                            macro_aps.append(average_precision_score(y_bin[:, i], y_proba[:, i]))
                        except Exception:
                            pass
                res["roc_curves_ovr"] = roc_curves if roc_curves else None
                res["pr_curves_ovr"] = pr_curves if pr_curves else None
                res["macro_auc"] = _safe_float(np.nanmean(macro_aucs)) if macro_aucs else None
                res["macro_ap"] = _safe_float(np.nanmean(macro_aps)) if macro_aps else None

                # Log loss (requires probabilities aligned with label order)
                try:
                    res["log_loss"] = _safe_float(log_loss(y_true, y_proba, labels=lbs))
                except Exception:
                    res["log_loss"] = None
        except Exception:
            # If anything goes wrong, keep deterministic keys
            res.setdefault("log_loss", None)
            res.setdefault("roc_curve", None)
            res.setdefault("pr_curve", None)
            res.setdefault("roc_curves_ovr", None)
            res.setdefault("pr_curves_ovr", None)
            res.setdefault("macro_auc", None)
            res.setdefault("macro_ap", None)

    return res


# ---------------------------
# Regression metrics
# ---------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE (guard zeros)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)) * 100.0
        mape = np.nanmean(ape)
    # SMAPE (symmetric)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = 100.0 * np.nanmean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return {
        "task": "regression",
        "mae": _safe_float(mae),
        "mse": _safe_float(mse),
        "rmse": _safe_float(rmse),
        "r2": _safe_float(r2),
        "mape": _safe_float(mape),
        "smape": _safe_float(smape),
    }


# ---------------------------
# Threshold sweep (binary)
# ---------------------------

def threshold_sweep_binary(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    thresholds: Optional[List[float]] = None,
    pos_label: Optional[Any] = None,
    average: str = "binary",
) -> Dict[str, Any]:
    """
    Compute precision/recall/f1 over a set of thresholds for a binary classifier.
    Returns JSON-serializable arrays for plotting threshold curves.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_proba_pos).reshape(-1)
    if thresholds is None:
        thresholds = list(np.linspace(0.0, 1.0, 101))
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        # map labels if y_true not in {0,1}
        if pos_label is not None and len(np.unique(y_true)) == 2 and set(np.unique(y_true)) != {0,1}:
            y_true_bin = (y_true == pos_label).astype(int)
        else:
            y_true_bin = y_true
        precs.append(_safe_float(precision_score(y_true_bin, y_pred, average="binary", zero_division=0)))
        recs.append(_safe_float(recall_score(y_true_bin, y_pred, average="binary", zero_division=0)))
        f1s.append(_safe_float(f1_score(y_true_bin, y_pred, average="binary", zero_division=0)))
    return {
        "thresholds": thresholds,
        "precision": precs,
        "recall": recs,
        "f1": f1s,
    }


# ---------------------------
# Convenience: end-to-end evaluators
# ---------------------------

def evaluate_supervised(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None,
    pos_label: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Auto-select metrics by task (classification vs regression) and compute a compact report.
    """
    task = infer_supervised_task(y_true)
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    ci = ClassifInputs(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_proba=None if y_proba is None else np.asarray(y_proba),
        labels=labels,
        average="weighted",
        pos_label=pos_label,
    )
    return classification_metrics(ci)
