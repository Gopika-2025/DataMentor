# src/modalities/image/profile.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

try:
    import pandas as pd  # optional for CSV mapping
except Exception:
    pd = None  # type: ignore

# -----------------------------
# Config / constants
# -----------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
DEFAULT_MAX_SCAN = 200_000         # directory scan cap
DEFAULT_META_SAMPLE = 5_000        # images to sample for width/height/mode
TINY_W, TINY_H = 32, 32            # flag very small images
SMALL_CLASS_THRESHOLD = 10         # warn when class count < this


# -----------------------------
# Utilities
# -----------------------------

def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXT


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def _pct(n: int, d: int) -> float:
    return float((n / d) * 100.0) if d else 0.0


@dataclass
class ImgMeta:
    width: int
    height: int
    mode: str


def _read_meta_fast(p: Path) -> Optional[ImgMeta]:
    """
    Safely open an image and read width/height/mode without decoding full pixels.
    Returns None if unreadable/corrupt.
    """
    try:
        with Image.open(p) as im:
            return ImgMeta(width=int(im.width), height=int(im.height), mode=str(im.mode))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


# -----------------------------
# Folder (ImageFolder) profiling
# -----------------------------

def _scan_images_in_folder(root: Path, max_items: int = DEFAULT_MAX_SCAN) -> List[Path]:
    """
    Recursively find images under 'root', capped by max_items.
    """
    root = Path(root)
    paths: List[Path] = []
    for i, p in enumerate(root.rglob("*")):
        if i >= max_items:
            break
        if _is_img(p):
            paths.append(p)
    return paths


def _infer_class_from_path(p: Path, root: Path) -> str:
    """
    Heuristic: class = immediate parent folder name under root.
    """
    try:
        return p.parent.relative_to(root).parts[0]
    except Exception:
        return p.parent.name


def _class_distribution(imgs: List[Path], root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in imgs:
        cls = _infer_class_from_path(p, root)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _sample_list(xs: List[Any], k: int) -> List[Any]:
    return xs[:k] if len(xs) <= k else xs[:k]


def _collect_meta_stats(paths: List[Path], root: Path, sample: int = DEFAULT_META_SAMPLE) -> Dict[str, Any]:
    """
    Read metadata for up to 'sample' images. Returns summary stats.
    """
    take = paths[:sample]
    metas: List[ImgMeta] = []
    corrupt = 0
    tiny = 0
    modes: Dict[str, int] = {}
    wh_list: List[Tuple[int, int]] = []

    for p in take:
        meta = _read_meta_fast(p)
        if meta is None:
            corrupt += 1
            continue
        metas.append(meta)
        wh_list.append((meta.width, meta.height))
        modes[meta.mode] = modes.get(meta.mode, 0) + 1
        if meta.width < TINY_W or meta.height < TINY_H:
            tiny += 1

    widths = [m.width for m in metas]
    heights = [m.height for m in metas]
    aspects = [(w / h) for (w, h) in wh_list if h > 0]

    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {}
        import numpy as np
        a = np.array(arr, dtype=float)
        return {
            "count": float(a.size),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=0)),
            "min": float(a.min()),
            "p50": float(np.quantile(a, 0.50)),
            "p90": float(np.quantile(a, 0.90)),
            "p99": float(np.quantile(a, 0.99)),
            "max": float(a.max()),
        }

    return {
        "sample_size": len(take),
        "readable_images": len(metas),
        "corrupt_in_sample": corrupt,
        "corrupt_in_sample_pct": _pct(corrupt, len(take)),
        "very_small_lt_32x32_in_sample": tiny,
        "width_stats": _stats([float(x) for x in widths]),
        "height_stats": _stats([float(x) for x in heights]),
        "aspect_ratio_stats": _stats([float(x) for x in aspects]),
        "mode_counts_top": dict(sorted(modes.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "sample_paths": [_rel(root, p) for p in _sample_list(paths, 8)],
    }


def profile_image_folder(root: Path) -> Dict[str, Any]:
    """
    Profile an ImageFolder-style dataset under 'root'.

    Returns JSON-serializable dict with:
      - n_images, n_classes, class_counts_top20, small_classes_lt10
      - integrity/meta: corrupt_in_sample, very_small count,
        width/height/aspect stats, mode histogram
      - sample_paths (relative)
    """
    root = Path(root)
    imgs = _scan_images_in_folder(root, max_items=DEFAULT_MAX_SCAN)
    n_imgs = len(imgs)

    class_counts = _class_distribution(imgs, root)
    n_classes = len(class_counts)
    small_classes = [c for c, n in class_counts.items() if n < SMALL_CLASS_THRESHOLD]
    top20_classes = dict(sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)[:20])

    meta_stats = _collect_meta_stats(imgs, root, sample=DEFAULT_META_SAMPLE)

    return {
        "dataset_type": "folder",
        "root": str(root),
        "n_images": int(n_imgs),
        "n_classes": int(n_classes),
        "class_counts_top20": top20_classes,
        "small_classes_lt10": small_classes[:50],
        "integrity_and_meta": meta_stats,
    }


# -----------------------------
# CSV mapping profiling
# -----------------------------

def profile_image_csv(
    csv_path: Path,
    image_col: str,
    label_col: Optional[str] = None,
    base_dir: Optional[Path] = None,
    max_rows_scan: int = 500_000,
    meta_sample: int = DEFAULT_META_SAMPLE,
) -> Dict[str, Any]:
    """
    Profile an image dataset defined by a CSV/Parquet/… with an image path column,
    and optional label column.
    """
    if pd is None:
        raise ImportError("pandas is required for profile_image_csv")

    csv_path = Path(csv_path)
    suf = csv_path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(csv_path)
    elif suf == ".parquet":
        df = pd.read_parquet(csv_path)
    elif suf == ".feather":
        df = pd.read_feather(csv_path)
    elif suf in {".xlsx", ".xls"}:
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    if len(df) > max_rows_scan:
        df = df.iloc[:max_rows_scan].copy()

    assert image_col in df.columns, f"Column '{image_col}' not found"

    def _resolve(p: str) -> Path:
        pth = Path(p)
        if base_dir is not None and not pth.is_absolute():
            return (Path(base_dir) / pth).resolve()
        return pth.resolve()

    paths: List[Path] = [_resolve(str(x)) for x in df[image_col].astype(str).tolist()]
    exists_flags = [int(p.exists()) for p in paths]
    n_exist = int(sum(exists_flags))
    n_total = int(len(paths))
    missing = n_total - n_exist

    class_counts: Dict[str, int] = {}
    if label_col and (label_col in df.columns):
        ser = df[label_col]
        vc = ser.value_counts(dropna=False).to_dict()
        class_counts = {
            ("NaN" if (isinstance(k, float) and math.isnan(k)) else str(k)): int(v)  # type: ignore
            for k, v in vc.items()
        }

    img_candidates = [p for p in paths if p.exists() and _is_img(p)]
    root_for_rel = (Path(base_dir) if base_dir else csv_path.parent)
    meta_stats = _collect_meta_stats(img_candidates, root_for_rel, sample=meta_sample)
    sample_paths = [_rel(root_for_rel, p) for p in img_candidates[:8]]

    small_classes = []
    if class_counts:
        small_classes = [c for c, n in class_counts.items() if n < SMALL_CLASS_THRESHOLD]

    return {
        "dataset_type": "csv_mapping",
        "table_path": str(csv_path),
        "base_dir": str(root_for_rel),
        "rows_scanned": n_total,
        "images_found": n_exist,
        "missing_paths": int(missing),
        "class_counts_top20": dict(sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]) if class_counts else {},
        "small_classes_lt10": small_classes[:50],
        "integrity_and_meta": {**meta_stats, "sample_paths": sample_paths},
    }


# -----------------------------
# Auto entry (folder or CSV)
# -----------------------------

def profile_image_dataset(
    source: Path,
    *,
    csv_image_col: Optional[str] = None,
    csv_label_col: Optional[str] = None,
    csv_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Auto profile an image dataset given a folder or a csv mapping.
    - If 'source' is a directory: use folder mode.
    - If 'source' is a file ending with .csv/.parquet/.feather/.xlsx: use csv mode (requires pandas).
    """
    source = Path(source)
    if source.is_dir():
        return profile_image_folder(source)
    if source.suffix.lower() in {".csv", ".parquet", ".feather", ".xlsx", ".xls"}:
        assert csv_image_col, "For CSV/Parquet mapping, 'csv_image_col' must be provided."
        return profile_image_csv(source, image_col=csv_image_col, label_col=csv_label_col, base_dir=csv_base_dir)
    raise ValueError(f"Unsupported source for image profiling: {source}")
