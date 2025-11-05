# src/modalities/tabular/preprocess.py
from __future__ import annotations
import src.core.preprocess as preprocess
# src/core/preprocess.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Union

import os, io, re, math, json
import numpy as np
import pandas as pd
from pathlib import Path

# Optional deps (safe-imports)
try:
    import cv2; HAS_CV2 = True
except Exception:
    HAS_CV2 = False
try:
    from PIL import Image, ImageOps, ImageFilter; HAS_PIL = True
except Exception:
    HAS_PIL = False
try:
    import librosa, soundfile as sf; HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False
try:
    import decord; HAS_DECORD = True
except Exception:
    HAS_DECORD = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
try:
    from bs4 import BeautifulSoup; HAS_BS4 = True
except Exception:
    HAS_BS4 = False
try:
    from sentence_transformers import SentenceTransformer; HAS_ST = True
except Exception:
    HAS_ST = False

# sklearn imports only (no self-imports)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, PowerTransformer,
    KBinsDiscretizer, PolynomialFeatures
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
