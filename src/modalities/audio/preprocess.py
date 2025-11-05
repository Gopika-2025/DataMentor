# src/modalities/audio/preprocess.py
from __future__ import annotations
"""
Audio preprocessing for DataMentor.

Exports:
- build_preprocessor(...) -> AudioPreprocessor
- load_and_features(pre, path_or_bytes) -> np.ndarray (mel-spectrogram or MFCC)

Backed by core:
  - src.core.preprocess.AudioPreprocessCfg
  - src.core.preprocess.AudioPreprocessor
  - src.core.preprocess.audio_preprocessor
"""

from typing import Union
from pathlib import Path
import numpy as np

# Soft import guard so we can surface a clear error if deps are missing
try:
    import librosa  # noqa: F401
    import soundfile as sf  # noqa: F401
    _AUDIO_DEPS_OK = True
except Exception:
    _AUDIO_DEPS_OK = False

from src.core.preprocess import (
    audio_preprocessor,
    AudioPreprocessCfg,
    AudioPreprocessor,
)


def build_preprocessor(
    *,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    trim_silence: bool = True,
    top_db: int = 30,
    features: str = "melspec",  # "melspec" | "mfcc"
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mfcc: int = 20,
    augment: bool = False,
    pitch_shift_steps: int = 0,
    time_stretch: float = 1.0,
) -> AudioPreprocessor:
    """
    Build an audio preprocessor (load → trim/augment/normalize → mel-spec/MFCC).
    """
    if not _AUDIO_DEPS_OK:
        raise ImportError(
            "Audio dependencies missing. Please install:\n"
            "  pip install librosa soundfile"
        )
    cfg = AudioPreprocessCfg(
        target_sr=target_sr,
        mono=mono,
        normalize=normalize,
        trim_silence=trim_silence,
        top_db=top_db,
        features=features,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        augment=augment,
        pitch_shift_steps=pitch_shift_steps,
        time_stretch=time_stretch,
    )
    return audio_preprocessor(cfg)


def load_and_features(pre: AudioPreprocessor, path_or_bytes: Union[str, Path, bytes]) -> np.ndarray:
    """
    Load audio then compute features using the pre's config.

    Parameters
    ----------
    pre : AudioPreprocessor
        Built via build_preprocessor(...)
    path_or_bytes : str | Path | bytes
        File path or in-memory bytes of an audio file.

    Returns
    -------
    np.ndarray (float32)
        - mel-spectrogram: shape (n_mels, T)
        - MFCC:            shape (n_mfcc, T)
    """
    if not _AUDIO_DEPS_OK:
        raise ImportError(
            "Audio dependencies missing. Please install:\n"
            "  pip install librosa soundfile"
        )
    y = pre.load(path_or_bytes)
    return pre.preprocess(y)


__all__ = ["build_preprocessor", "load_and_features"]
