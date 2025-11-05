# src/config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable


class Paths:
    """
    Centralized project paths.
    - Works no matter where you run `streamlit run app.py` from.
    - Auto-creates data/, artifacts/, reports/ and modality subfolders.
    Usage:
        from src.config import paths
        paths.data, paths.artifacts, paths.reports
        paths.ensure()  # idempotent
    """

    def __init__(self) -> None:
        # Project root = repo root (…/DataMentor)
        # src/config.py -> parents[1] == <repo_root>/src/.. -> <repo_root>
        self._root = Path(__file__).resolve().parents[1]

        # Default modality names used across the project
        self._modalities = ("tabular", "text", "image", "audio", "video", "figures")

        # Make sure base dirs exist on import
        self.ensure()

    # ---- Base folders ----
    @property
    def root(self) -> Path:
        return self._root

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    # ---- Modality helpers ----
    @property
    def modalities(self) -> Iterable[str]:
        return self._modalities

    def modality_dir(self, base: Path, name: str) -> Path:
        """
        Return (and ensure) a modality-specific subfolder under `base`.
        Example:
            paths.modality_dir(paths.artifacts, "text") -> <root>/artifacts/text
        """
        p = base / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def all_modality_dirs(self) -> Dict[str, Dict[str, Path]]:
        """
        Convenience: returns a mapping of {modality: {artifacts: Path, reports: Path}}
        and ensures they exist.
        """
        out: Dict[str, Dict[str, Path]] = {}
        for m in self.modalities:
            out[m] = {
                "artifacts": self.modality_dir(self.artifacts, m),
                "reports": self.modality_dir(self.reports, m),
            }
        return out

    # ---- Ensurers ----
    def ensure(self) -> None:
        """Ensure base and modality subfolders exist."""
        # Base dirs
        for p in (self.data, self.artifacts, self.reports):
            p.mkdir(parents=True, exist_ok=True)

        # Common sub-structure
        self.all_modality_dirs()

        # Data/raw and data/processed are used frequently
        (self.data / "raw").mkdir(parents=True, exist_ok=True)
        (self.data / "processed").mkdir(parents=True, exist_ok=True)


# Singleton used across the app
paths = Paths()
