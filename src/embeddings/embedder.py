"""
Embedding generation via OpenAI text-embedding models.
Supports batch processing with configurable batch size.
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger
from openai import OpenAI


_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Embedder:
    """Generate vector embeddings using OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model     = model
        self.dimension = _DIMENSIONS.get(model, 1536)
        self._client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"Embedder initialised: model={model} dim={self.dimension}")

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string — used for query embedding."""
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def embed_batch(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """
        Embed a list of strings in batches.
        Returns embeddings in the same order as inputs.
        """
        embeddings: list[list[float]] = []
        total_batches = (len(texts) - 1) // batch_size + 1

        for i in range(0, len(texts), batch_size):
            batch   = texts[i : i + batch_size]
            batch_n = i // batch_size + 1
            logger.info(f"Embedding batch {batch_n}/{total_batches} ({len(batch)} texts)")
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )
            embeddings.extend(item.embedding for item in response.data)

        logger.info(f"Embedded {len(embeddings)} texts total")
        return embeddings
