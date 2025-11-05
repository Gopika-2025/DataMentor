# src/modalities/tabular/profile.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------- Helpers ---------

def _safe_pct(n: int, d: int) -> float:
    return float((n / d) * 100.0) if d else 0.0


def _as_str_dtype(ser: pd.Series) -> str:
    # Normalize pandas dtypes to a friendly string
    dt = ser.dtype
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime64"
    if pd.api.types.is_timedelta64_dtype(dt):
        return "timedelta64"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    return "object"


def _sample_df(df: pd.DataFrame, max_rows: int = 200000, random_state: int = 42) -> pd.DataFrame:
    """Downsample rows for heavy operations to keep profiling snappy."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def _quantiles(ser: pd.Series, qs: Iterable[float]) -> Dict[str, float]:
    q = ser.quantile(list(qs), interpolation="linear")
    out = {f"p{int(k*100)}": float(v) for k, v in q.items()}
    return out


# --------- Core sections ---------

def schema_summary(df: pd.DataFrame) -> Dict[str, Any]:
    dtypes = {c: _as_str_dtype(df[c]) for c in df.columns}
    nulls = {c: int(df[c].isna().sum()) for c in df.columns}
    nrows, ncols = df.shape
    nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    # heuristic: very-high cardinality categoricals (for encoding choice)
    high_card = [c for c in df.columns if nunique[c] > max(50, int(0.2 * nrows))]
    return {
        "rows": int(nrows),
        "cols": int(ncols),
        "dtypes": dtypes,
        "null_counts": nulls,
        "nunique": nunique,
        "high_cardinality_candidates": high_card[:50],
    }


def memory_summary(df: pd.DataFrame) -> Dict[str, Any]:
    total = float(df.memory_usage(deep=True).sum())
    per_col = df.memory_usage(deep=True).to_dict()
    per_col = {str(k): float(v) for k, v in per_col.items() if k != "Index"}
    return {"total_bytes": total, "per_column_bytes": per_col}


def duplicate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    nrows = len(df)
    dup_mask = df.duplicated(keep="first")
    n_dups = int(dup_mask.sum())
    return {
        "duplicate_rows": n_dups,
        "duplicate_rows_pct": _safe_pct(n_dups, nrows),
    }


def datetime_overview(df: pd.DataFrame) -> Dict[str, Any]:
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    info: Dict[str, Any] = {"datetime_columns": dt_cols}
    for c in dt_cols[:25]:
        s = df[c].dropna()
        if s.empty:
            rng = {"min": None, "max": None}
        else:
            rng = {"min": s.min().isoformat(), "max": s.max().isoformat()}
        info[f"{c}__range"] = rng
    return info


def numeric_stats(df: pd.DataFrame, sample_rows: int = 200000) -> Dict[str, Any]:
    sdf = _sample_df(df, sample_rows)
    num = sdf.select_dtypes(include=["number"])
    if num.empty:
        return {
            "n_numeric": 0,
            "columns": {},
            "pearson_corr_top": [],
        }

    columns: Dict[str, Any] = {}
    for c in num.columns[:200]:  # cap to avoid huge payloads
        s = num[c].dropna()
        if s.empty:
            columns[c] = {"count": 0}
            continue
        desc = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
        desc.update(_quantiles(s, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
        # simple outlier hint via IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = float(q3 - q1)
        low_cut = float(q1 - 1.5 * iqr)
        high_cut = float(q3 - 1.5 * iqr)
        out_low = int((s < low_cut).sum())
        out_high = int((s > high_cut).sum())
        desc.update({
            "iqr": iqr,
            "outliers_low": out_low,
            "outliers_high": out_high,
            "outliers_pct": _safe_pct(out_low + out_high, len(s)),
        })
        columns[c] = desc

    # lightweight correlation overview (top absolute pairs)
    try:
        corr = num.corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, 0.0)
        # pick top 20 pairs
        stacked = (
            corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
                .stack()
                .sort_values(ascending=False)
        )
        top_pairs = [
            {"col_a": str(a), "col_b": str(b), "abs_corr": float(v)}
            for (a, b), v in stacked.head(20).items()
        ]
    except Exception:
        top_pairs = []

    return {
        "n_numeric": int(num.shape[1]),
        "columns": columns,
        "pearson_corr_top": top_pairs,
    }


def categorical_stats(df: pd.DataFrame, max_categories_preview: int = 30, sample_rows: int = 200000) -> Dict[str, Any]:
    sdf = _sample_df(df, sample_rows)
    cat = sdf.select_dtypes(exclude=["number", "datetime", "timedelta", "bool"]).copy()
    # Also include boolean as categorical for preview
    for c in sdf.columns:
        if pd.api.types.is_bool_dtype(sdf[c]):
            cat[c] = sdf[c]

    columns: Dict[str, Any] = {}
    for c in [col for col in cat.columns if col in sdf.columns][:200]:
        s = sdf[c].astype("object")
        nun = int(s.nunique(dropna=True))
        vc = s.value_counts(dropna=False).head(max_categories_preview)
        vc = {("NaN" if (isinstance(k, float) and math.isnan(k)) else str(k)): int(v) for k, v in vc.items()}

        columns[c] = {
            "nunique": nun,
            "top_values": vc,
        }
    return {
        "n_categorical": int(len(columns)),
        "columns": columns,
    }


def missingness_summary(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    per_col = {c: {"missing": int(df[c].isna().sum()), "missing_pct": _safe_pct(int(df[c].isna().sum()), n)}
               for c in df.columns}
    # rows with ANY missing
    rows_any = int(df.isna().any(axis=1).sum())
    return {
        "rows_with_any_missing": rows_any,
        "rows_with_any_missing_pct": _safe_pct(rows_any, n),
        "per_column": per_col,
    }


def target_preview(df: pd.DataFrame, target: Optional[str]) -> Dict[str, Any]:
    if not target or target not in df.columns:
        return {"target": None}
    s = df[target]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 15:
        # regression preview
        s = s.dropna()
        out: Dict[str, Any] = {
            "target": target,
            "task": "regression",
            "count": int(s.shape[0]),
            "mean": float(s.mean()) if not s.empty else None,
            "std": float(s.std(ddof=0)) if not s.empty else None,
        }
        if not s.empty:
            out.update(_quantiles(s, [0.05, 0.5, 0.95]))
        return out
    # classification preview
    vc = s.value_counts(dropna=False)
    top = vc.head(50)
    minority = int(vc.min()) if not vc.empty else 0
    majority = int(vc.max()) if not vc.empty else 0
    imb = float(majority / minority) if minority else None
    return {
        "target": target,
        "task": "classification",
        "n_classes": int(s.nunique(dropna=True)),
        "class_counts_top50": {("NaN" if (isinstance(k, float) and math.isnan(k)) else str(k)): int(v) for k, v in top.items()},
        "imbalance_ratio_majority_over_minority": imb,
    }


# --------- Public entrypoint ---------

def profile_tabular(
    df: pd.DataFrame,
    target: Optional[str] = None,
    sample_rows_numeric: int = 200000,
    max_categories_preview: int = 30,
) -> Dict[str, Any]:
    """
    Produce a compact, JSON-serializable profile for tabular datasets.

    Sections:
      - schema_summary: rows/cols, dtypes, null counts, nunique, high-cardinality hints
      - memory_summary: bytes per column + total
      - duplicate_summary: duplicate rows and %
      - datetime_overview: min/max per datetime col
      - numeric_stats: mean/std/min/max/quantiles, IQR outliers, top correlated pairs
      - categorical_stats: nunique + top values per categorical
      - missingness_summary: per-column missingness and overall row missingness
      - target_preview: task inference + class counts (or regression summary)

    Heavy operations downsample rows to keep it responsive on large data.
    """
    prof: Dict[str, Any] = {}
    try:
        prof["schema_summary"] = schema_summary(df)
    except Exception as e:
        prof["schema_summary_error"] = str(e)

    try:
        prof["memory_summary"] = memory_summary(df)
    except Exception as e:
        prof["memory_summary_error"] = str(e)

    try:
        prof["duplicate_summary"] = duplicate_summary(df)
    except Exception as e:
        prof["duplicate_summary_error"] = str(e)

    try:
        prof["datetime_overview"] = datetime_overview(df)
    except Exception as e:
        prof["datetime_overview_error"] = str(e)

    try:
        prof["numeric_stats"] = numeric_stats(df, sample_rows=sample_rows_numeric)
    except Exception as e:
        prof["numeric_stats_error"] = str(e)

    try:
        prof["categorical_stats"] = categorical_stats(df, max_categories_preview=max_categories_preview)
    except Exception as e:
        prof["categorical_stats_error"] = str(e)

    try:
        prof["missingness_summary"] = missingness_summary(df)
    except Exception as e:
        prof["missingness_summary_error"] = str(e)

    try:
        prof["target_preview"] = target_preview(df, target)
    except Exception as e:
        prof["target_preview_error"] = str(e)

    return prof
