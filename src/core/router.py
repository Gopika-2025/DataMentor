# src/core/router.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal, Optional, Set, Tuple
import zipfile

# ---------------------------
# Supported extensions by type
# ---------------------------
TABULAR_EXT: Set[str] = {".csv", ".parquet", ".feather", ".xlsx", ".xls", ".jsonl"}
TEXT_EXT:    Set[str] = {".txt", ".jsonl"}  # text can also live inside tabular files
IMAGE_EXT:   Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
AUDIO_EXT:   Set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXT:   Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

Modality = Literal["tabular", "text", "image", "audio", "video", "unknown"]

# Order matters below (confidence priority during detection)
_DETECTION_ORDER: Tuple[Set[str], Modality] = (
    # Will be used differently; see decide_from_exts()
)

def _ext(path: Path) -> str:
    """Get lowercase suffix, normalized."""
    return path.suffix.lower()

def _is_zip(path: Path) -> bool:
    return _ext(path) == ".zip"

# ---------------------------
# Sniffing helpers
# ---------------------------
def sniff_exts_in_dir(folder: Path, sample: int = 300) -> Set[str]:
    """
    Collect a sample of file extensions in a directory tree.
    """
    exts: Set[str] = set()
    count = 0
    for p in folder.rglob("*"):
        if p.is_file():
            exts.add(_ext(p))
            count += 1
            if count >= sample:
                break
    return exts

def sniff_exts_in_zip(zip_path: Path, sample: int = 300) -> Set[str]:
    """
    Collect a sample of file extensions inside a zip file (without extracting).
    """
    exts: Set[str] = set()
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            # Skip directories
            if info.is_dir():
                continue
            exts.add(Path(info.filename).suffix.lower())
            count += 1
            if count >= sample:
                break
    return exts

# ---------------------------
# Decision logic
# ---------------------------
def decide_from_exts(exts: Set[str]) -> Modality:
    """
    Decide modality from a set of extensions.
    Priority:
      1) If any tabular ext present -> tabular
      2) Else image
      3) Else audio
      4) Else video
      5) Else text
      6) Else unknown
    Rationale: CSV/Parquet/Excel are high-signal that the user intends tabular.
    """
    if exts & TABULAR_EXT:
        return "tabular"
    if exts & IMAGE_EXT:
        return "image"
    if exts & AUDIO_EXT:
        return "audio"
    if exts & VIDEO_EXT:
        return "video"
    if exts & TEXT_EXT:
        return "text"
    return "unknown"

def detect_modality(path: Path | str, ui_hint: Optional[Modality | Literal["auto"]] = None) -> Modality:
    """
    Detect the data modality from a single file (CSV, JPG, MP3, MP4, etc),
    a directory (of media files), or a .zip containing a tree of files.

    Parameters
    ----------
    path : Path | str
        Path to file or directory (or .zip).
    ui_hint : Optional[Modality | Literal["auto"]]
        If provided and not "auto", returns this value directly, allowing the UI to override.

    Returns
    -------
    Modality
        One of {"tabular", "text", "image", "audio", "video", "unknown"}.
    """
    if ui_hint and ui_hint != "auto":
        return ui_hint  # explicit user override

    p = Path(path)
    if not p.exists():
        return "unknown"

    # Folder: sniff a subset of files
    if p.is_dir():
        exts = sniff_exts_in_dir(p)
        return decide_from_exts(exts)

    # Single file
    if _is_zip(p):
        exts = sniff_exts_in_zip(p)
        return decide_from_exts(exts)

    return decide_from_exts({ _ext(p) })

# ---------------------------
# Query helpers
# ---------------------------
def is_tabular_file(path: Path) -> bool:
    return _ext(path) in TABULAR_EXT

def is_text_file(path: Path) -> bool:
    return _ext(path) in TEXT_EXT

def is_image_file(path: Path) -> bool:
    return _ext(path) in IMAGE_EXT

def is_audio_file(path: Path) -> bool:
    return _ext(path) in AUDIO_EXT

def is_video_file(path: Path) -> bool:
    return _ext(path) in VIDEO_EXT

def modality_for_ext(ext: str) -> Modality:
    if ext in TABULAR_EXT: return "tabular"
    if ext in IMAGE_EXT:   return "image"
    if ext in AUDIO_EXT:   return "audio"
    if ext in VIDEO_EXT:   return "video"
    if ext in TEXT_EXT:    return "text"
    return "unknown"

def list_files_by_modality(root: Path | str, modality: Modality, limit: Optional[int] = None) -> List[Path]:
    """
    List files under a folder (or inside a zip) that match the requested modality.

    For ZIP inputs, we return pseudo-paths like Path("zip://file.zip#inner/path.jpg")
    only for *listing* purposes; extraction should be handled by the caller if needed.

    For regular folders, returns real filesystem paths.

    Parameters
    ----------
    root : Path | str
        Directory or ZIP file.
    modality : Modality
        Desired modality to filter.
    limit : Optional[int]
        Optional maximum number of files to return.

    Returns
    -------
    List[Path]
    """
    root = Path(root)
    files: List[Path] = []

    def _maybe_add(p: Path):
        nonlocal files
        if modality_for_ext(_ext(p)) == modality:
            files.append(p)

    if not root.exists():
        return files

    if root.is_dir():
        count = 0
        for p in root.rglob("*"):
            if p.is_file():
                _maybe_add(p)
                count += 1
                if limit and len(files) >= limit:
                    break
        return files

    if _is_zip(root):
        # Represent zip members as synthetic paths: zip://<zip>#<inner>
        count = 0
        with zipfile.ZipFile(root) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                inner = Path(info.filename)
                if modality_for_ext(inner.suffix.lower()) == modality:
                    files.append(Path(f"zip://{root.as_posix()}#{inner.as_posix()}"))
                count += 1
                if limit and len(files) >= limit:
                    break
        return files

    # single regular file
    if modality_for_ext(_ext(root)) == modality:
        return [root]
    return []

def describe_modality(modality: Modality) -> str:
    """Short human description for UI badges/messages."""
    return {
        "tabular": "Tabular data (CSV/Parquet/Excel/JSONL)",
        "text":    "Unstructured text files",
        "image":   "Images (JPG/PNG/WEBP/TIFF)",
        "audio":   "Audio (WAV/MP3/FLAC/OGG/M4A)",
        "video":   "Video (MP4/MOV/AVI/MKV/WEBM)",
        "unknown": "Unknown / mixed content",
    }[modality]


