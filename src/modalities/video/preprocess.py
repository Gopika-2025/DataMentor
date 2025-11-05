# src/modalities/image/preprocess.py
from __future__ import annotations
"""
Image preprocessing for DataMentor.

This module handles:
  - Reading image files or arrays
  - Resizing / cropping / padding
  - Color conversion (RGB ↔ Grayscale)
  - Normalization (0–1, mean/std)
  - Optional enhancements: CLAHE, histogram equalization, denoise, sharpen, edge
"""

from typing import List, Union, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from src.core.preprocess import (
    image_preprocessor,
    ImagePreprocessCfg,
    ImagePreprocessor,
)


def build_preprocessor(
    *,
    target_size: Tuple[int, int] = (224, 224),
    keep_aspect: bool = True,
    center_crop: bool = False,
    grayscale: bool = False,
    normalize: bool = True,
    mean_std_norm: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
    clahe: bool = False,
    equalize_hist: bool = False,
    denoise: bool = False,
    sharpen: bool = False,
    edge: bool = False,
    pad: Optional[int] = None,
) -> ImagePreprocessor:
    """
    Build an image preprocessor configuration object.

    Parameters
    ----------
    target_size : tuple of (H, W)
        Desired output size.
    keep_aspect : bool
        Maintain aspect ratio when resizing.
    center_crop : bool
        Crop from the image center before resize.
    grayscale : bool
        Convert to grayscale (1-channel).
    normalize : bool
        Scale pixel values to [0,1].
    mean_std_norm : (mean, std)
        Apply mean/std normalization (per channel).
    clahe : bool
        Apply adaptive histogram equalization.
    equalize_hist : bool
        Apply global histogram equalization.
    denoise : bool
        Apply OpenCV fastNlMeansDenoisingColored.
    sharpen : bool
        Apply sharpening kernel.
    edge : bool
        Apply Sobel edge detection.
    pad : int
        Optional padding pixels around image.
    """
    cfg = ImagePreprocessCfg(
        target_size=target_size,
        keep_aspect=keep_aspect,
        center_crop=center_crop,
        grayscale=grayscale,
        normalize=normalize,
        mean_std_norm=mean_std_norm,
        clahe=clahe,
        equalize_hist=equalize_hist,
        denoise=denoise,
        sharpen=sharpen,
        edge=edge,
        pad=pad,
    )
    return image_preprocessor(cfg)


def transform_batch(
    pre: ImagePreprocessor,
    batch: List[Union[str, Path, np.ndarray]],
) -> List[np.ndarray]:
    """
    Transform a list of images (paths or np.ndarray) into model-ready arrays.

    Parameters
    ----------
    pre : ImagePreprocessor
        Built from build_preprocessor().
    batch : list
        List of paths or already loaded np.ndarray images.

    Returns
    -------
    list of np.ndarray
        Preprocessed image arrays (float32, HxWxC).
    """
    return pre.transform_batch(batch)


# ---- Local fallback implementation (if src/core/preprocess.py not loaded) ---- #
if not HAS_CV2:
    def _simple_resize(img: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        raise ImportError("cv2 not installed — please install opencv-python.")
else:
    def _simple_resize(img: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        return cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_AREA)
