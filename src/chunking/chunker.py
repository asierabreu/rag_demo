"""
Text chunking stage.
Splits page/row documents into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter, preserving all source metadata.
Each chunk gets a deterministic ID derived from document name + position.
"""
from __future__ import annotations

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class DocumentChunker:
    """Split document pages/rows into fixed-size overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Accept a list of document dicts (from DocumentLoader) and return a
        flat list of chunk dicts, each with 'id', 'text', and 'metadata'.
        """
        chunks: list[dict] = []
        for doc in documents:
            raw_chunks = self._splitter.split_text(doc["text"])
            doc_key    = doc["metadata"].get("page", doc["metadata"].get("row", 0))

            for i, text in enumerate(raw_chunks):
                text = text.strip()
                if not text:
                    continue
                chunks.append({
                    "id":   self._make_id(doc["metadata"]["document_name"], doc_key, i),
                    "text": text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index":  i,
                        "total_chunks": len(raw_chunks),
                    },
                })

        logger.info(
            f"Chunking complete: {len(chunks)} chunks from {len(documents)} pages/rows "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(doc_name: str, position: Any, chunk_idx: int) -> str:
        """Deterministic MD5 chunk ID — safe for Pinecone vector IDs."""
        raw = f"{doc_name}|{position}|{chunk_idx}"
        return hashlib.md5(raw.encode()).hexdigest()
