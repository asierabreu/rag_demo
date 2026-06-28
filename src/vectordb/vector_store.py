"""
Pinecone vector store: index management, upsert, query, and deletion.
"""
from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger
from pinecone import Pinecone, ServerlessSpec


class VectorStore:
    """Manages all interactions with the Pinecone vector index."""

    UPSERT_BATCH = 100          # Pinecone recommended upsert batch size
    METADATA_TEXT_LIMIT = 1000  # chars stored alongside each vector

    def __init__(
        self,
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine",
    ) -> None:
        self.index_name = index_name
        self.dimension  = dimension
        self.metric     = metric
        self._pc        = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self._index     = self._get_or_create_index()

    # ── Index lifecycle ────────────────────────────────────────────────────

    def _get_or_create_index(self):
        existing = {i.name for i in self._pc.list_indexes()}
        if self.index_name not in existing:
            logger.info(f"Creating Pinecone index '{self.index_name}' dim={self.dimension}")
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait until the index is ready
            while not self._pc.describe_index(self.index_name).status["ready"]:
                logger.debug("Waiting for index to be ready …")
                time.sleep(2)
            logger.info("Pinecone index ready")
        else:
            index_info = self._pc.describe_index(self.index_name)
            existing_dimension = getattr(index_info, "dimension", None)
            if existing_dimension is not None and existing_dimension != self.dimension:
                raise RuntimeError(
                    f"Pinecone index '{self.index_name}' has dimension {existing_dimension}, "
                    f"but the configured embedding model produces {self.dimension}. "
                    "Delete and recreate the index, or point config.yaml to a matching index "
                    "before uploading documents."
                )
            logger.info(f"Using existing Pinecone index '{self.index_name}'")
        return self._pc.Index(self.index_name)

    # ── Write ──────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        namespace: str = "default",
    ) -> int:
        """Upsert chunk embeddings + metadata into Pinecone."""
        vectors = [
            {
                "id":     chunk["id"],
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    # Store truncated text for display in the UI
                    "text": chunk["text"][: self.METADATA_TEXT_LIMIT],
                },
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        total = 0
        for i in range(0, len(vectors), self.UPSERT_BATCH):
            batch = vectors[i : i + self.UPSERT_BATCH]
            self._index.upsert(vectors=batch, namespace=namespace)
            total += len(batch)
            logger.info(f"Upserted {total}/{len(vectors)} vectors to namespace '{namespace}'")

        return total

    # ── Read ───────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        namespace: str = "default",
        filter_mission: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most similar vectors."""
        pinecone_filter = (
            {"mission_name": {"$eq": filter_mission}} if filter_mission else None
        )
        result = self._index.query(
            vector=embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=pinecone_filter,
        )
        return [
            {
                "id":       match.id,
                "score":    match.score,
                "metadata": match.metadata,
                "text":     match.metadata.get("text", ""),
            }
            for match in result.matches
        ]

    # ── Delete ─────────────────────────────────────────────────────────────

    def delete_by_mission(self, mission_name: str, namespace: str = "default") -> None:
        """Delete all vectors tagged with a given mission."""
        self._index.delete(
            filter={"mission_name": {"$eq": mission_name}},
            namespace=namespace,
        )
        logger.info(f"Deleted vectors for mission='{mission_name}' namespace='{namespace}'")

    def delete_by_document(self, document_name: str, namespace: str = "default") -> None:
        """Delete all vectors from a specific document."""
        self._index.delete(
            filter={"document_name": {"$eq": document_name}},
            namespace=namespace,
        )
        logger.info(f"Deleted vectors for document='{document_name}'")

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_stats(self, namespace: str = "default") -> dict[str, Any]:
        stats = self._index.describe_index_stats()
        return stats.to_dict()
