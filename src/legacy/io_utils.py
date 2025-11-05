# D:\project\DataMentor\src\legacy\io_utils.py
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import pandas as pd

try:
    import pyarrow  # noqa: F401  # for parquet/feather via pandas
except Exception:
    pass


PathLike = Union[str, Path]

# ================================================================
# 🔧 Path & Safe Write Utilities
# ================================================================

def ensure_dir(path: PathLike) -> Path:
    """Create directory if it doesn’t exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: PathLike, data: str, encoding: str = "utf-8") -> None:
    """Write text atomically — avoids corruption on crash."""
    p = Path(path)
    ensure_dir(p.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent), encoding=encoding) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, p)


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Atomic write for binary data."""
    p = Path(path)
    ensure_dir(p.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(p.parent)) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, p)


def file_exists(path: PathLike) -> bool:
    """Check if a file exists."""
    return Path(path).exists()


# ================================================================
# 📄 JSON / JSONL Helpers
# ================================================================

def read_json(path: PathLike, default: Any = None) -> Any:
    """Read JSON file or return default if not found."""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: PathLike, *, indent: int = 2, ensure_ascii: bool = False) -> Path:
    """Save JSON with pretty indentation."""
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    atomic_write_text(path, text, encoding="utf-8")
    return Path(path)


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Stream JSON Lines (NDJSON)."""
    p = Path(path)
    if not p.exists():
        return iter(())
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, Any]], path: PathLike) -> Path:
    """Write iterable of dicts as JSONL."""
    buf = io.StringIO()
    for rec in records:
        buf.write(json.dumps(rec, ensure_ascii=False))
        buf.write("\n")
    atomic_write_text(path, buf.getvalue(), encoding="utf-8")
    return Path(path)


# ================================================================
# 🧮 CSV Delimiter Detection
# ================================================================

_COMMON_DELIMS = [",", ";", "\t", "|", ":"]

def sniff_delimiter(path: PathLike, sample_bytes: int = 64_000) -> str:
    """Detect CSV delimiter from sample bytes."""
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(sample_bytes)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_COMMON_DELIMS))
        return dialect.delimiter
    except Exception:
        counts = {d: sample.count(d) for d in _COMMON_DELIMS}
        return max(counts, key=counts.get)


# ================================================================
# 📊 Table I/O (CSV, Excel, JSONL, Parquet, Feather)
# ================================================================

def read_table(path: PathLike, *, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read a tabular file automatically by extension."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        delim = sniff_delimiter(p)
        return pd.read_csv(p, sep=delim)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet) if sheet else pd.read_excel(p)
    if suf == ".jsonl":
        return pd.DataFrame(iter_jsonl(p))
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".feather":
        return pd.read_feather(p)
    if suf == ".json":
        return pd.DataFrame(read_json(p))
    raise ValueError(f"Unsupported table format: {suf} (path={p})")


def write_table(df: pd.DataFrame, path: PathLike, *, index: bool = False) -> Path:
    """Write a DataFrame to CSV, Excel, JSONL, or Parquet."""
    p = Path(path)
    ensure_dir(p.parent)
    suf = p.suffix.lower()
    if suf == ".csv":
        df.to_csv(p, index=index)
    elif suf in {".xlsx", ".xls"}:
        df.to_excel(p, index=index)
    elif suf == ".jsonl":
        records = df.to_dict(orient="records")
        write_jsonl(records, p)
    elif suf == ".parquet":
        df.to_parquet(p, index=index)
    elif suf == ".feather":
        df.to_feather(p)
    elif suf == ".json":
        write_json(df.to_dict(orient="records"), p)
    else:
        raise ValueError(f"Unsupported save format: {suf} (path={p})")
    return p


def read_csv_in_chunks(path: PathLike, *, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    """Stream large CSV in chunks."""
    delim = sniff_delimiter(path)
    for chunk in pd.read_csv(path, sep=delim, chunksize=chunksize):
        yield chunk


def concat_tables(paths: List[PathLike]) -> pd.DataFrame:
    """Combine multiple tables vertically."""
    parts = [read_table(p) for p in paths]
    return pd.concat(parts, ignore_index=True)


# ================================================================
# 🧠 Type Inference Helpers
# ================================================================

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

def infer_and_coerce_dtypes(df: pd.DataFrame, *, try_datetime: bool = True) -> pd.DataFrame:
    """Coerce object columns to numeric/datetime where possible."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_object_dtype(s):
            # numeric
            if s.dropna().astype(str).str.fullmatch(_NUM_RE).mean() > 0.9:
                out[c] = pd.to_numeric(s, errors="coerce")
                continue
            # datetime
            if try_datetime or ("date" in c.lower()) or ("time" in c.lower()):
                try:
                    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        out[c] = parsed
                except Exception:
                    pass
    return out


# ================================================================
# 🔐 Hash / Checksum Utilities
# ================================================================

def sha256_of_file(path: PathLike, *, chunk: int = 1 << 20) -> str:
    """Return SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_of_bytes(b: bytes) -> str:
    """Return SHA256 checksum of byte string."""
    return hashlib.sha256(b).hexdigest()


# ================================================================
# 🗂️ ZIP Extraction Utilities
# ================================================================

def extract_zip(zip_path: PathLike, dest_dir: PathLike, *, overwrite: bool = True) -> Path:
    """Extract ZIP archive into dest_dir/<zipname>/."""
    z = Path(zip_path)
    dest = Path(dest_dir)
    ensure_dir(dest)
    target = dest / z.stem
    if overwrite and target.exists():
        for p in target.rglob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            target.rmdir()
        except Exception:
            pass
    ensure_dir(target)
    with zipfile.ZipFile(z, "r") as zipf:
        zipf.extractall(target)
    return target


# ================================================================
# 📝 Text Read / Write
# ================================================================

def read_text(path: PathLike, *, encoding: str = "utf-8", default: Optional[str] = None) -> Optional[str]:
    """Read UTF-8 text file."""
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding=encoding)


def write_text(text: str, path: PathLike, *, encoding: str = "utf-8") -> Path:
    """Write UTF-8 text atomically."""
    atomic_write_text(path, text, encoding=encoding)
    return Path(path)


# ================================================================
# 📦 Public Aliases (for backward compatibility)
# ================================================================

load_json = read_json
save_json = write_json
load_table = read_table
save_table = write_table
