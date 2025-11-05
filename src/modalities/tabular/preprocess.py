# src/modalities/tabular/preprocess.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler


# ---------------------------
# Dataclass config
# ---------------------------

@dataclass
class PreprocessConfig:
    # Columns (None = auto-detect from df)
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None

    # Strategies
    num_impute: str = "median"          # "mean" | "median" | "most_frequent" | "constant"
    cat_impute: str = "most_frequent"   # "most_frequent" | "constant"

    # Scaling
    scaler: str = "standard"            # "standard" | "robust" | "none"

    # Encoding
    one_hot_drop: Optional[str] = None  # None | "first" | "if_binary"
    handle_unknown: str = "ignore"      # "ignore" | "error"
    max_ohe_categories: int = 200       # safety cap for huge cardinality

    # Heuristics
    high_card_threshold: int = 50       # > this = considered high-card categorical
    treat_bool_as_categorical: bool = True

    # Misc
    target: Optional[str] = None        # not transformed; just kept for reference


# ---------------------------
# Presets you can select from
# ---------------------------

preset_cfg: Dict[str, PreprocessConfig] = {
    # Good general baseline
    "baseline": PreprocessConfig(
        num_impute="median",
        cat_impute="most_frequent",
        scaler="standard",
        one_hot_drop=None,
        handle_unknown="ignore",
        max_ohe_categories=200,
        high_card_threshold=50,
    ),
    # More robust to outliers
    "robust": PreprocessConfig(
        num_impute="median",
        cat_impute="most_frequent",
        scaler="robust",
        one_hot_drop=None,
        handle_unknown="ignore",
        max_ohe_categories=200,
        high_card_threshold=50,
    ),
    # Minimal transform (useful for tree models)
    "minimal": PreprocessConfig(
        num_impute="median",
        cat_impute="most_frequent",
        scaler="none",
        one_hot_drop=None,
        handle_unknown="ignore",
        max_ohe_categories=300,
        high_card_threshold=100,
    ),
}


def make_cfg(preset: str = "baseline", **overrides) -> PreprocessConfig:
    """Get a config from a preset with optional field overrides."""
    base = preset_cfg.get(preset, preset_cfg["baseline"])
    data = asdict(base)
    data.update(overrides or {})
    return PreprocessConfig(**data)


# ---------------------------
# Column auto-detection
# ---------------------------

def _auto_detect_columns(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    if cfg.target and cfg.target in cols:
        cols.remove(cfg.target)

    # numeric
    num_cols = list(df[cols].select_dtypes(include=["number"]).columns)

    # categoricals
    cat_cols = [c for c in cols if c not in num_cols]

    # keep booleans as categorical if requested
    if not cfg.treat_bool_as_categorical:
        bool_cols = [c for c in df[cols].select_dtypes(include=["bool"]).columns]
        cat_cols = [c for c in cat_cols if c not in bool_cols]
        num_cols += bool_cols  # treat bool as numeric 0/1

    # filter extremely high cardinality categoricals from OHE (kept, but truncated later)
    nunique = {c: int(df[c].nunique(dropna=True)) for c in cat_cols}
    # we won't drop them, but we will cap OHE categories below via OneHotEncoder params
    # just return the lists
    return num_cols, cat_cols


# ---------------------------
# Build sklearn preprocessors
# ---------------------------

def _numeric_pipeline(cfg: PreprocessConfig) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy=cfg.num_impute))]
    if cfg.scaler == "standard":
        steps.append(("scale", StandardScaler()))
    elif cfg.scaler == "robust":
        steps.append(("scale", RobustScaler()))
    # else "none": no scaler
    return Pipeline(steps)


def _categorical_pipeline(cfg: PreprocessConfig) -> Pipeline:
    # Impute
    cat_imputer = SimpleImputer(strategy=cfg.cat_impute, fill_value="__MISSING__")

    # One-hot with caps and safe unknown handling
    ohe = OneHotEncoder(
        handle_unknown=cfg.handle_unknown,
        drop=cfg.one_hot_drop,
        sparse=True,
        max_categories=cfg.max_ohe_categories,  # sklearn>=1.1
    )
    return Pipeline([("impute", cat_imputer), ("ohe", ohe)])


def build_preprocessor(
    df: pd.DataFrame,
    cfg: PreprocessConfig | None = None,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer given dataframe + config.
    Returns: (preprocessor, numeric_cols, categorical_cols)
    """
    cfg = cfg or make_cfg("baseline")

    # Columns
    if cfg.numeric_cols is None or cfg.categorical_cols is None:
        num_cols, cat_cols = _auto_detect_columns(df, cfg)
        if cfg.numeric_cols is None:
            cfg.numeric_cols = num_cols
        if cfg.categorical_cols is None:
            cfg.categorical_cols = cat_cols

    # Pipelines
    num_pipe = _numeric_pipeline(cfg) if cfg.numeric_cols else "drop"
    cat_pipe = _categorical_pipeline(cfg) if cfg.categorical_cols else "drop"

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, cfg.numeric_cols or []),
            ("cat", cat_pipe, cfg.categorical_cols or []),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # prefer sparse output when many OHE features
        n_jobs=None,
        verbose_feature_names_out=True,
    )
    return pre, (cfg.numeric_cols or []), (cfg.categorical_cols or [])
