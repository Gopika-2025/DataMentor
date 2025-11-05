# D:\project\DataMentor\utils\io.py
from __future__ import annotations

import io
import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import pandas as pd

try:
    import joblib  # optional, for pickle support
except Exception:
    joblib = None


# -------------------------------------------------------------
# Type alias
# -------------------------------------------------------------
PathLike = Union[str, Path]


# =============================================================
# 🗂️ Folder Utilities
# =============================================================

def ensure_dir(path: PathLike) -> Path:
    """Create directory (and parents) if missing."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_exists(path: PathLike) -> bool:
    """Check if a file exists."""
    return Path(path).exists()


def list_files_by_ext(root: PathLike, exts: Iterable[str], recursive: bool = True) -> List[Path]:
    """Return list of files under a directory with given extensions."""
    root = Path(root)
    exts_l = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    it = root.rglob("*") if recursive else root.glob("*")
    return [p for p in it if p.is_file() and p.suffix.lower() in exts_l]


# =============================================================
# 💾 Atomic Write Helpers
# =============================================================

def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """Atomic write for text files (avoids corruption)."""
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


# =============================================================
# 🧾 JSON / JSONL
# =============================================================

def save_json(obj: Any, path: PathLike, *, indent: int = 2, ensure_ascii: bool = False) -> Path:
    """Save Python object as JSON (pretty printed)."""
    p = Path(path)
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    _atomic_write_text(p, text, encoding="utf-8")
    return p


def load_json(path: PathLike, *, default: Any = None) -> Any:
    """Load JSON if exists, else return default."""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[Dict[str, Any]], path: PathLike) -> Path:
    """Write iterable of dicts as JSON Lines."""
    p = Path(path)
    ensure_dir(p.parent)
    buf = io.StringIO()
    for rec in records:
        buf.write(json.dumps(rec, ensure_ascii=False))
        buf.write("\n")
    _atomic_write_text(p, buf.getvalue(), encoding="utf-8")
    return p


def read_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Read JSON Lines file as generator of dicts."""
    p = Path(path)
    if not p.exists():
        return iter(())
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# =============================================================
# 📊 Table I/O (CSV / Excel / Parquet / Feather / JSONL)
# =============================================================

def load_table(path: PathLike) -> pd.DataFrame:
    """
    Load tabular data automatically from CSV, Excel, Parquet, Feather, or JSONL.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".feather":
        return pd.read_feather(p)
    if suf == ".jsonl":
        rows = list(read_jsonl(p))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported table format: {suf} (path={p})")


def save_table(df: pd.DataFrame, path: PathLike, *, index: bool = False) -> Path:
    """
    Save DataFrame to supported formats (CSV, Excel, Parquet, Feather, JSONL).
    """
    p = Path(path)
    ensure_dir(p.parent)
    suf = p.suffix.lower()
    if suf == ".csv":
        df.to_csv(p, index=index)
    elif suf in {".xlsx", ".xls"}:
        df.to_excel(p, index=index)
    elif suf == ".parquet":
        df.to_parquet(p, index=index)
    elif suf == ".feather":
        df.to_feather(p)
    elif suf == ".jsonl":
        write_jsonl(df.to_dict(orient="records"), p)
    else:
        raise ValueError(f"Unsupported save format: {suf} (path={p})")
    return p


# =============================================================
# 🧱 Pickle / Joblib
# =============================================================

def save_pickle(obj: Any, path: PathLike) -> Path:
    """Serialize Python object using joblib or pickle."""
    p = Path(path)
    ensure_dir(p.parent)
    if joblib is not None:
        joblib.dump(obj, p)
    else:
        import pickle
        with p.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_pickle(path: PathLike) -> Any:
    """Load serialized object from pickle/joblib file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if joblib is not None:
        return joblib.load(p)
    import pickle
    with p.open("rb") as f:
        return pickle.load(f)


# =============================================================
# 📝 Text Read / Write
# =============================================================

def save_text(text: str, path: PathLike, *, encoding: str = "utf-8") -> Path:
    """Save text file with atomic write."""
    p = Path(path)
    _atomic_write_text(p, text, encoding=encoding)
    return p


def load_text(path: PathLike, *, encoding: str = "utf-8", default: Optional[str] = None) -> Optional[str]:
    """Read text file safely."""
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding=encoding)
