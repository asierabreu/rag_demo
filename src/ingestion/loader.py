"""
Document loaders for PDF and CSV files.
Returns a list of page/row dicts ready for the chunking stage.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import pypdf
from loguru import logger


class DocumentLoader:
    """Load PDF and CSV files into raw text pages/rows with metadata."""

    SUPPORTED = {".pdf", ".csv"}

    # ── Public API ─────────────────────────────────────────────────────────

    def load(self, file_path: str, mission_name: str) -> list[dict[str, Any]]:
        """Load a document from disk path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        self._check_ext(path.suffix)
        logger.info(f"Loading {path.suffix.upper()} document: {path.name} | mission={mission_name}")
        return self._dispatch(path.suffix, open(path, "rb").read(), path.name, mission_name)

    def load_from_bytes(
        self, file_bytes: bytes, filename: str, mission_name: str
    ) -> list[dict[str, Any]]:
        """Load a document from raw bytes (file-upload endpoint)."""
        ext = Path(filename).suffix.lower()
        self._check_ext(ext)
        logger.info(f"Loading {ext.upper()} from bytes: {filename} | mission={mission_name}")
        return self._dispatch(ext, file_bytes, filename, mission_name)

    # ── Dispatching ────────────────────────────────────────────────────────

    def _dispatch(
        self, ext: str, data: bytes, filename: str, mission_name: str
    ) -> list[dict[str, Any]]:
        if ext == ".pdf":
            return self._load_pdf(data, filename, mission_name)
        return self._load_csv(data, filename, mission_name)

    # ── PDF ────────────────────────────────────────────────────────────────

    def _load_pdf(
        self, data: bytes, filename: str, mission_name: str
    ) -> list[dict[str, Any]]:
        reader = pypdf.PdfReader(io.BytesIO(data))
        total = len(reader.pages)
        docs: list[dict] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            docs.append({
                "text": text,
                "metadata": {
                    "source_type":    "pdf",
                    "document_name":  filename,
                    "mission_name":   mission_name,
                    "page":           i + 1,
                    "total_pages":    total,
                },
            })
        logger.info(f"PDF loaded: {len(docs)}/{total} non-empty pages from {filename}")
        return docs

    # ── CSV ────────────────────────────────────────────────────────────────

    def _load_csv(
        self, data: bytes, filename: str, mission_name: str
    ) -> list[dict[str, Any]]:
        df = pd.read_csv(io.BytesIO(data))
        cols = list(df.columns)
        docs: list[dict] = []
        for idx, row in df.iterrows():
            # Serialise each row as "column: value | column: value …"
            text = " | ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if pd.notna(val) and str(val).strip()
            )
            if not text:
                continue
            docs.append({
                "text": text,
                "metadata": {
                    "source_type":   "csv",
                    "document_name": filename,
                    "mission_name":  mission_name,
                    "row":           int(idx) + 1,
                    "columns":       cols,
                },
            })
        logger.info(f"CSV loaded: {len(docs)} rows from {filename}")
        return docs

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _check_ext(ext: str) -> None:
        if ext.lower() not in DocumentLoader.SUPPORTED:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {DocumentLoader.SUPPORTED}"
            )
