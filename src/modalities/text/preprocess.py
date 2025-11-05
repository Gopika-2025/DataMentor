# src/modalities/text/preprocess.py
from __future__ import annotations

"""
Text preprocessing for DataMentor.

This wraps the shared builders in `src/core/preprocess.py` and exposes:
- build_preprocessor(...): returns a scikit-learn Pipeline that cleans text and
  vectorizes using TF-IDF / Bag-of-Words / Sentence-Transformer embeddings.
- make_cfg(...): convenience factory to construct a TextPreprocessCfg from UI toggles.

Usage:
    from src.modalities.text.preprocess import build_preprocessor, make_cfg

    cfg = make_cfg(vectorizer="tfidf", ngrams=(1,2), max_features=50000,
                   stopwords=True, lemmatize=True, stemming=False)
    text_pre = build_preprocessor(**cfg)  # or build_preprocessor(vectorizer="tfidf", ...)
    X = text_pre.fit_transform(list_of_strings)
"""

from typing import Optional, Tuple, Dict, Any
from sklearn.pipeline import Pipeline

from src.core.preprocess import text_pipeline, TextPreprocessCfg


def build_preprocessor(
    *,
    vectorizer: str = "tfidf",            # "tfidf" | "bow" | "st_embeddings"
    ngrams: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = 50000,
    # cleaning toggles
    lowercase: bool = True,
    strip_punct: bool = True,
    strip_numbers: bool = False,
    strip_extra_spaces: bool = True,
    remove_urls_emails: bool = True,
    remove_html: bool = True,
    stopwords: bool = True,
    stemming: bool = False,
    lemmatize: bool = True,
    contractions: bool = True,
    unicode_norm: bool = True,
    language: str = "english",
) -> Pipeline:
    """
    Build a text preprocessing pipeline (clean → vectorize).

    Parameters
    ----------
    vectorizer : {"tfidf","bow","st_embeddings"}
        TF-IDF, Bag-of-Words, or Sentence-Transformer embeddings.
        For "st_embeddings", install `sentence-transformers`.
    ngrams : (min_n, max_n)
        N-gram range for tfidf/bow.
    max_features : int or None
        Cap features for tfidf/bow; None keeps all.
    cleaning toggles : bool
        Controls normalization, HTML/URL removal, stopwords, stemming, lemmatization, etc.
    language : str
        Stopword language (if NLTK stopwords is available).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline that transforms a list/Series of strings into features.
    """
    cfg = TextPreprocessCfg(
        vectorizer=vectorizer,
        ngrams=ngrams,
        max_features=max_features,
        lowercase=lowercase,
        strip_punct=strip_punct,
        strip_numbers=strip_numbers,
        strip_extra_spaces=strip_extra_spaces,
        remove_urls_emails=remove_urls_emails,
        remove_html=remove_html,
        stopwords=stopwords,
        stemming=stemming,
        lemmatize=lemmatize,
        contractions=contractions,
        unicode_norm=unicode_norm,
        language=language,
    )
    return text_pipeline(cfg)


def make_cfg(
    *,
    vectorizer: str = "tfidf",            # "tfidf" | "bow" | "st_embeddings"
    ngrams: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = 50000,
    lowercase: bool = True,
    strip_punct: bool = True,
    strip_numbers: bool = False,
    strip_extra_spaces: bool = True,
    remove_urls_emails: bool = True,
    remove_html: bool = True,
    stopwords: bool = True,
    stemming: bool = False,
    lemmatize: bool = True,
    contractions: bool = True,
    unicode_norm: bool = True,
    language: str = "english",
) -> Dict[str, Any]:
    """
    Convenience factory that returns kwargs for build_preprocessor(...).
    (Useful when wiring from Streamlit UI toggles.)
    """
    return dict(
        vectorizer=vectorizer,
        ngrams=ngrams,
        max_features=max_features,
        lowercase=lowercase,
        strip_punct=strip_punct,
        strip_numbers=strip_numbers,
        strip_extra_spaces=strip_extra_spaces,
        remove_urls_emails=remove_urls_emails,
        remove_html=remove_html,
        stopwords=stopwords,
        stemming=stemming,
        lemmatize=lemmatize,
        contractions=contractions,
        unicode_norm=unicode_norm,
        language=language,
    )
