"""
Embedding generation via pluggable providers.
Supports OpenAI text embeddings and local sentence-transformers models.
"""
from __future__ import annotations

import importlib
import os
from typing import Any

from loguru import logger


OpenAI = None


_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Embedder:
    """Generate vector embeddings using the configured provider."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.dimension = _DIMENSIONS.get(model, 1536)
        self._client: Any = None

        if self.provider == "openai":
            from openai import OpenAI as OpenAIClient

            self._client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider in {"sentence-transformers", "local", "hf"}:
            # Load the local model lazily on first use so app startup stays fast.
            self._client = None
        else:
            raise ValueError(
                f"Unknown embedding provider '{provider}'. "
                "Supported providers: openai, sentence-transformers"
            )

        logger.info(
            f"Embedder initialised: provider={self.provider} model={self.model} dim={self.dimension}"
        )

    def _load_local_model(self):
        if self._client is None:
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._client = sentence_transformers.SentenceTransformer(self.model)
        return self._client

    def _ensure_client(self):
        if self.provider == "openai":
            return self._client
        return self._load_local_model()

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string — used for query embedding."""
        if self.provider == "openai":
            response = self._client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding

        client = self._ensure_client()
        embedding = client.encode(text, normalize_embeddings=True)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def embed_batch(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """
        Embed a list of strings in batches.
        Returns embeddings in the same order as inputs.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        total_batches = (len(texts) - 1) // batch_size + 1

        if self.provider != "openai":
            client = self._ensure_client()
            encoded = client.encode(texts, normalize_embeddings=True, batch_size=batch_size)
            for row in encoded:
                embeddings.append(row.tolist() if hasattr(row, "tolist") else list(row))
            logger.info(f"Embedded {len(embeddings)} texts total")
            return embeddings

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
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