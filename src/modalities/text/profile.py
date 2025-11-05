# Text profiling 
# src/modalities/text/profile.py
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple

import math
import re
import numpy as np
import pandas as pd


# ---------------------------
# Internal helpers
# ---------------------------

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?")  # basic Latin + accents
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def _safe_pct(n: int, d: int) -> float:
    return float((n / d) * 100.0) if d else 0.0

def _quantiles(arr: pd.Series, qs: Iterable[float]) -> Dict[str, float]:
    q = arr.quantile(list(qs), interpolation="linear")
    return {f"p{int(k*100)}": float(v) for k, v in q.items()}

def _tokenize_words(text: str) -> list[str]:
    # Very lightweight tokenizer (no external deps)
    return _WORD_RE.findall(text.lower())

def _bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:])) if len(tokens) >= 2 else []

def _char_comp_metrics(s: pd.Series) -> Dict[str, Any]:
    # Character composition quick check (digits / punctuation / uppercase ratio per text)
    if len(s) == 0:
        return {"digits_pct_mean": 0.0, "punct_pct_mean": 0.0, "upper_pct_mean": 0.0}

    lengths = s.str.len().replace(0, np.nan)

    digits = s.str.count(r"\d")
    punct = s.apply(lambda t: len(_PUNCT_RE.findall(t)) if isinstance(t, str) else 0)
    upper = s.str.count(r"[A-Z]")

    # Avoid division by zero
    digits_pct = (digits / lengths).fillna(0.0) * 100.0
    punct_pct  = (punct  / lengths).fillna(0.0) * 100.0
    upper_pct  = (upper  / lengths).fillna(0.0) * 100.0

    return {
        "digits_pct_mean": float(digits_pct.mean()),
        "punct_pct_mean": float(punct_pct.mean()),
        "upper_pct_mean": float(upper_pct.mean()),
    }

# ---------------------------
# Public API
# ---------------------------

def profile_text(
    df: pd.DataFrame,
    text_col: str,
    target_col: Optional[str] = None,
    *,
    max_rows_for_token_stats: int = 100_000,
    top_k_tokens: int = 30,
    include_bigrams: bool = True,
) -> Dict[str, Any]:
    """
    Build a compact, JSON-serializable profile for a text dataset contained in a table.

    Parameters
    ----------
    df : DataFrame
        Input data.
    text_col : str
        Column name containing raw text.
    target_col : Optional[str]
        Optional label/target column for class distribution preview.
    max_rows_for_token_stats : int
        Cap rows used for token frequency stats (for speed).
    top_k_tokens : int
        Number of unigrams/bigrams to return.
    include_bigrams : bool
        Whether to compute bigram frequencies.

    Returns
    -------
    dict
        A dictionary safe to dump as JSON, with keys:
          - rows, text_col, target_col
          - missing_text, empty_text, whitespace_only
          - char_lengths: {count, mean, std, min, p50, p90, p99, max}
          - word_lengths: {count, mean, std, min, p50, p90, p99, max}
          - token_stats: {vocab_estimate, top_unigrams, top_bigrams?}
          - label_preview: {n_classes, class_counts_topK, imbalance_ratio}
          - per_label_text_len: {mean_chars_by_label, mean_words_by_label}
          - duplicates: {exact_duplicate_rows, duplicate_texts, top_duplicate_texts}
          - char_composition: {digits_pct_mean, punct_pct_mean, upper_pct_mean}
    """
    assert text_col in df.columns, f"Text column '{text_col}' not found."

    # ---- Basic counts
    n_rows = int(len(df))
    s = df[text_col].astype(str)  # coerce everything to string for safety

    missing_text = int(df[text_col].isna().sum())
    empty_text = int((s == "").sum())
    whitespace_only = int(s.str.fullmatch(r"\s*", na=False).sum())

    # ---- Length stats (characters)
    char_len = s.str.len()
    char_stats = {
        "count": int(char_len.count()),
        "mean": float(char_len.mean()),
        "std": float(char_len.std(ddof=0)),
        "min": int(char_len.min()) if len(char_len) else 0,
        "max": int(char_len.max()) if len(char_len) else 0,
    }
    char_stats.update(_quantiles(char_len, [0.5, 0.9, 0.99]))

    # ---- Tokenization for word length stats (sample if huge)
    if n_rows > max_rows_for_token_stats:
        s_tokens = s.sample(n=max_rows_for_token_stats, random_state=42)
    else:
        s_tokens = s

    token_lists = s_tokens.apply(_tokenize_words)
    word_counts = token_lists.apply(len)
    word_stats = {
        "count": int(word_counts.count()),
        "mean": float(word_counts.mean()),
        "std": float(word_counts.std(ddof=0)),
        "min": int(word_counts.min()) if len(word_counts) else 0,
        "max": int(word_counts.max()) if len(word_counts) else 0,
    }
    word_stats.update(_quantiles(word_counts, [0.5, 0.9, 0.99]))

    # ---- Token frequencies (unigrams + optional bigrams)
    unigram_ctr: Counter[str] = Counter()
    bigram_ctr: Counter[tuple[str, str]] = Counter()

    for toks in token_lists:
        if toks:
            unigram_ctr.update(toks)
            if include_bigrams:
                bigram_ctr.update(_bigrams(toks))

    top_unigrams = unigram_ctr.most_common(top_k_tokens)
    if include_bigrams:
        top_bigrams = [(" ".join(bg), cnt) for bg, cnt in bigram_ctr.most_common(top_k_tokens)]
    else:
        top_bigrams = []

    vocab_estimate = int(len(unigram_ctr))  # crude vocab size

    # ---- Label preview (if available)
    label_preview: Dict[str, Any] = {"target_col": None}
    mean_chars_by_label: Dict[str, float] = {}
    mean_words_by_label: Dict[str, float] = {}

    if target_col and target_col in df.columns:
        y = df[target_col]
        vc = y.value_counts(dropna=False)
        topK = vc.head(50).to_dict()
        if not vc.empty:
            minority = int(vc.min())
            majority = int(vc.max())
            imb = float(majority / minority) if minority else None
        else:
            imb = None

        # Per-label averages
        # (Use original s / token_lists aligned indexes)
        # Recompute tokens for full df when needed for accurate per-label stats
        tokens_full = s.apply(_tokenize_words)
        words_full = tokens_full.apply(len)

        mean_chars_by_label = (
            pd.DataFrame({"y": y, "chars": s.str.len()})
            .groupby("y", dropna=False)["chars"]
            .mean()
            .fillna(0.0)
            .to_dict()
        )
        # Convert NaN keys to string for JSON safety
        mean_chars_by_label = {("NaN" if (isinstance(k, float) and math.isnan(k)) else str(k)): float(v)
                               for k, v in mean_chars_by_label.items()}

        mean_words_by_label = (
            pd.DataFrame({"y": y, "words": words_full})
            .groupby("y", dropna=False)["words"]
            .mean()
            .fillna(0.0)
            .to_dict()
        )
        mean_words_by_label = {("NaN" if (isinstance(k, float) and math.isnan(k)) else str(k)): float(v)
                               for k, v in mean_words_by_label.items()}

        # Safe JSON conversion for class counts (NaN keys)
        class_counts_json = {
            ("NaN" if (isinstance(k, float) and math.isnan(k)) else str(k)): int(v)
            for k, v in topK.items()
        }

        label_preview = {
            "target_col": target_col,
            "n_classes": int(y.nunique(dropna=True)),
            "class_counts_top50": class_counts_json,
            "imbalance_ratio_majority_over_minority": imb,
        }

    # ---- Duplicates
    dup_rows = int(df.duplicated(keep="first").sum())
    # Duplicate texts (same text appears multiple times)
    text_counts = s.value_counts()
    dup_texts = int((text_counts > 1).sum())
    top_dup_texts = [
        {"text": str(idx)[:120], "count": int(cnt)}
        for idx, cnt in text_counts.head(10).items()
        if cnt > 1
    ]

    # ---- Character composition metrics
    char_comp = _char_comp_metrics(s)

    # ---- Assemble report
    report: Dict[str, Any] = {
        "rows": n_rows,
        "text_col": text_col,
        "target_col": target_col if (target_col in df.columns) else None,
        "missing_text": missing_text,
        "empty_text": empty_text,
        "whitespace_only": whitespace_only,
        "char_lengths": char_stats,
        "word_lengths": word_stats,
        "token_stats": {
            "vocab_estimate": vocab_estimate,
            "top_unigrams": top_unigrams,            # list[[token, count], ...]
            "top_bigrams": top_bigrams,              # list[[token token, count], ...]
        },
        "label_preview": label_preview,
        "per_label_text_len": {
            "mean_chars_by_label": mean_chars_by_label,
            "mean_words_by_label": mean_words_by_label,
        },
        "duplicates": {
            "exact_duplicate_rows": dup_rows,
            "duplicate_texts": dup_texts,
            "top_duplicate_texts": top_dup_texts,
        },
        "char_composition": char_comp,
    }
    return report
