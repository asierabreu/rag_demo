"""
Retrieval module.
Embeds a user query, fetches the top-k chunks from Pinecone,
and formats them into a context string for the LLM.
"""
from __future__ import annotations

from typing import Any

from loguru import logger


class Retriever:
    """Query encoder + context formatter for the RAG pipeline."""

    def __init__(
        self,
        embedder,
        vector_store,
        top_k: int = 5,
        score_threshold: float = 0.65,
    ) -> None:
        self.embedder        = embedder
        self.vector_store    = vector_store
        self.top_k           = top_k
        self.score_threshold = score_threshold

    # ── Public API ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        mission_name: str | None = None,
        namespace: str = "default",
    ) -> list[dict[str, Any]]:
        """
        Embed the query and return chunks that pass the score threshold.
        Optionally filter by mission name.
        """
        logger.info(
            f"Query: '{query[:80]}{'…' if len(query)>80 else ''}' "
            f"| mission={mission_name or 'all'}"
        )
        embedding = self.embedder.embed_text(query)
        results   = self.vector_store.query(
            embedding=embedding,
            top_k=self.top_k,
            namespace=namespace,
            filter_mission=mission_name,
        )
        filtered = [r for r in results if r["score"] >= self.score_threshold]
        logger.info(
            f"Retrieved {len(filtered)}/{len(results)} chunks "
            f"(threshold={self.score_threshold})"
        )
        return filtered

    def format_context(self, chunks: list[dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a numbered context block for the LLM.
        Each entry shows the source reference, score, and chunk text.
        """
        if not chunks:
            return "No relevant documentation found in the corpus."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta  = chunk["metadata"]
            doc   = meta.get("document_name", "Unknown document")
            miss  = meta.get("mission_name",  "Unknown mission")
            page  = meta.get("page", meta.get("row", "—"))
            score = chunk["score"]
            parts.append(
                f"[{i}] {doc} | Mission: {miss} | Page/Row: {page} "
                f"| Relevance: {score:.2f}\n{chunk['text']}"
            )

        return "\n\n---\n\n".join(parts)
