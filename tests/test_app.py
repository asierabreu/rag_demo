"""
Test suite for the ESA Ground Segment RAG application.

Run with:
    pytest tests/ -v
    pytest tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import io
import json
import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Minimal env so imports don't crash ────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY",    "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("GOOGLE_API_KEY",    "test-google-key")
os.environ.setdefault("PINECONE_API_KEY",  "test-pinecone-key")


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def mock_config() -> dict[str, Any]:
    return {
        "pinecone":   {"index_name": "test-index", "dimension": 1536,
                       "metric": "cosine", "namespace": "test-ns"},
        "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
        "chunking":   {"chunk_size": 500, "chunk_overlap": 50},
        "llm": {
            "default_provider": "anthropic",
            "providers": {
                "openai":    {"model": "gpt-4o",          "max_tokens": 512, "temperature": 0.0},
                "anthropic": {"model": "claude-sonnet-4-6", "max_tokens": 512, "temperature": 0.0},
                "google":    {"model": "gemini-1.5-pro",  "max_tokens": 512, "temperature": 0.0},
            },
        },
        "retrieval":  {"top_k": 3, "score_threshold": 0.5},
        "api":        {"host": "0.0.0.0", "port": 8000,
                       "title": "Test RAG", "version": "0.0.1"},
    }


@pytest.fixture(scope="session")
def client(mock_config):
    """FastAPI TestClient with all external services mocked."""
    with (
        patch("src.vectordb.vector_store.Pinecone"),
        patch("src.embeddings.embedder.OpenAI"),
    ):
        from src.api.routes import create_app
        app = create_app(mock_config)
        yield TestClient(app)


# ── Helpers ────────────────────────────────────────────────────────────────

def _fake_chunks(n: int = 2) -> list[dict]:
    return [
        {
            "id":    f"chunk-{i}",
            "score": 0.85 - i * 0.05,
            "text":  f"Ground segment documentation excerpt {i}. "
                     "The PLATO MCS uplink subsystem uses CCSDS TC packets.",
            "metadata": {
                "document_name": f"PLATO_MCS_ICD_v{i}.pdf",
                "mission_name":  "PLATO",
                "page":          i + 1,
                "source_type":   "pdf",
            },
        }
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════════
# DocumentLoader tests
# ══════════════════════════════════════════════════════════════════════════

class TestDocumentLoader:

    def test_load_pdf_bytes(self):
        from src.ingestion.loader import DocumentLoader

        # Build a minimal single-page PDF in memory with pypdf
        import pypdf
        writer = pypdf.PdfWriter()
        page   = writer.add_blank_page(width=612, height=792)
        buf    = io.BytesIO()
        writer.write(buf)

        loader = DocumentLoader()
        # Blank pages have no extractable text — expect empty list
        docs = loader.load_from_bytes(buf.getvalue(), "test.pdf", "TestMission")
        assert isinstance(docs, list)

    def test_load_csv_bytes(self):
        from src.ingestion.loader import DocumentLoader

        csv_data = b"system,status,document\nMCS,operational,ICD-001\nMPS,planned,SRS-002\n"
        loader   = DocumentLoader()
        docs     = loader.load_from_bytes(csv_data, "systems.csv", "PLATO")

        assert len(docs) == 2
        assert docs[0]["metadata"]["source_type"]  == "csv"
        assert docs[0]["metadata"]["mission_name"] == "PLATO"
        assert "MCS" in docs[0]["text"]
        assert "ICD-001" in docs[0]["text"]

    def test_unsupported_format_raises(self):
        from src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load_from_bytes(b"data", "report.docx", "Gaia")

    def test_missing_file_raises(self):
        from src.ingestion.loader import DocumentLoader

        with pytest.raises(FileNotFoundError):
            DocumentLoader().load("/nonexistent/path/doc.pdf", "Gaia")


# ══════════════════════════════════════════════════════════════════════════
# DocumentChunker tests
# ══════════════════════════════════════════════════════════════════════════

class TestDocumentChunker:

    def test_chunk_produces_output(self):
        from src.chunking.chunker import DocumentChunker

        long_text = ("The Mission Control System interfaces with the ground station. " * 30)
        docs = [{"text": long_text,
                 "metadata": {"document_name": "MCS_SRS.pdf",
                              "mission_name": "PLATO", "page": 1,
                              "source_type": "pdf"}}]
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks  = chunker.chunk(docs)

        assert len(chunks) > 1
        for c in chunks:
            assert "id"       in c
            assert "text"     in c
            assert "metadata" in c
            assert len(c["text"]) <= 220   # small tolerance

    def test_chunk_ids_are_unique(self):
        from src.chunking.chunker import DocumentChunker

        text = "A" * 1000
        docs = [{"text": text,
                 "metadata": {"document_name": "doc.pdf",
                              "mission_name": "Gaia", "page": 1,
                              "source_type": "pdf"}}]
        chunks = DocumentChunker(chunk_size=100, chunk_overlap=0).chunk(docs)
        ids    = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_metadata_preserved(self):
        from src.chunking.chunker import DocumentChunker

        docs = [{"text": "Short document text for metadata test.",
                 "metadata": {"document_name": "ICD.pdf",
                              "mission_name": "CHEOPS", "page": 3,
                              "source_type": "pdf"}}]
        chunks = DocumentChunker().chunk(docs)
        assert chunks[0]["metadata"]["mission_name"]  == "CHEOPS"
        assert chunks[0]["metadata"]["document_name"] == "ICD.pdf"
        assert chunks[0]["metadata"]["page"]          == 3


# ══════════════════════════════════════════════════════════════════════════
# Retriever tests
# ══════════════════════════════════════════════════════════════════════════

class TestRetriever:

    def _make_retriever(self, chunks: list[dict]):
        from src.retrieval.retriever import Retriever

        embedder     = MagicMock()
        vector_store = MagicMock()
        embedder.embed_text.return_value     = [0.1] * 1536
        vector_store.query.return_value      = chunks
        return Retriever(embedder, vector_store, top_k=3, score_threshold=0.6)

    def test_retrieve_filters_by_threshold(self):
        chunks = [
            {"id": "a", "score": 0.9, "text": "High relevance chunk.", "metadata": {}},
            {"id": "b", "score": 0.4, "text": "Low relevance chunk.",  "metadata": {}},
        ]
        retriever = self._make_retriever(chunks)
        results   = retriever.retrieve("test query")
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_format_context_numbered(self):
        from src.retrieval.retriever import Retriever

        retriever = Retriever(MagicMock(), MagicMock())
        chunks    = _fake_chunks(2)
        context   = retriever.format_context(chunks)
        assert "[1]" in context
        assert "[2]" in context
        assert "PLATO" in context

    def test_format_context_empty(self):
        from src.retrieval.retriever import Retriever

        retriever = Retriever(MagicMock(), MagicMock())
        result    = retriever.format_context([])
        assert "No relevant" in result


# ══════════════════════════════════════════════════════════════════════════
# LLM Factory tests
# ══════════════════════════════════════════════════════════════════════════

class TestLLMFactory:

    def test_available_providers(self):
        from src.llm.llm_client import LLMFactory

        providers = LLMFactory.available_providers()
        assert "openai"    in providers
        assert "anthropic" in providers
        assert "google"    in providers
        assert "ollama"    in providers

    def test_unknown_provider_raises(self):
        from src.llm.llm_client import LLMFactory

        with pytest.raises(ValueError, match="Unknown provider"):
            LLMFactory.create("mistral")

    def test_create_anthropic(self):
        from src.llm.llm_client import AnthropicClient, LLMFactory

        with patch("anthropic.Anthropic"):
            client = LLMFactory.create("anthropic", {"model": "claude-sonnet-4-6"})
            assert isinstance(client, AnthropicClient)
            assert client.provider_name == "anthropic"

    def test_create_openai(self):
        from src.llm.llm_client import LLMFactory, OpenAIClient

        with patch("openai.OpenAI"):
            client = LLMFactory.create("openai", {"model": "gpt-4o"})
            assert isinstance(client, OpenAIClient)
            assert client.provider_name == "openai"


class TestEmbedder:

    def test_sentence_transformers_provider(self):
        fake_module = types.ModuleType("sentence_transformers")

        class FakeSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def get_sentence_embedding_dimension(self) -> int:
                return 384

            def encode(self, texts, normalize_embeddings=True, batch_size=None):
                if isinstance(texts, str):
                    return [0.1, 0.2, 0.3]
                return [[0.1, 0.2, 0.3] for _ in texts]

        fake_module.SentenceTransformer = FakeSentenceTransformer

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            from src.embeddings.embedder import Embedder

            embedder = Embedder(provider="sentence-transformers", model="all-MiniLM-L6-v2")
            assert embedder.dimension == 384
            assert len(embedder.embed_text("hello world")) == 3
            assert len(embedder.embed_batch(["a", "b"])) == 2


# ══════════════════════════════════════════════════════════════════════════
# Prompt template tests
# ══════════════════════════════════════════════════════════════════════════

class TestPromptTemplates:

    def test_build_rag_prompt_contains_query(self):
        from src.prompts.prompt_templates import build_rag_prompt

        prompt = build_rag_prompt("What is the uplink rate?", "some context", "")
        assert "What is the uplink rate?" in prompt
        assert "some context"             in prompt

    def test_no_context_response_contains_query(self):
        from src.prompts.prompt_templates import build_no_context_response

        msg = build_no_context_response("What is the MPS schedule format?", "PLATO")
        assert "MPS schedule format" in msg
        assert "PLATO"               in msg

    def test_system_prompt_mentions_esa(self):
        from src.prompts.prompt_templates import SYSTEM_PROMPT

        assert "ESA"   in SYSTEM_PROMPT
        assert "MCS"   in SYSTEM_PROMPT
        assert "ECSS"  in SYSTEM_PROMPT


# ══════════════════════════════════════════════════════════════════════════
# API endpoint tests (TestClient)
# ══════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        with patch("src.vectordb.vector_store.VectorStore.get_stats",
                   return_value={"total_vector_count": 42, "dimension": 1536}):
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status"              in data
        assert "available_providers" in data


class TestChatEndpoint:

    def test_empty_query_returns_400(self, client):
        resp = client.post("/api/chat", json={"query": "  ", "provider": "anthropic"})
        assert resp.status_code == 400

    def test_chat_returns_answer(self, client):
        with (
            patch("src.retrieval.retriever.Retriever.retrieve",
                  return_value=_fake_chunks(2)),
            patch("src.llm.llm_client.AnthropicClient.chat",
                  return_value="The PLATO uplink uses CCSDS TC packets at 4 kbps."),
        ):
            resp = client.post("/api/chat", json={
                "query":    "What is the PLATO uplink rate?",
                "provider": "anthropic",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer"           in data
        assert "sources"          in data
        assert "session_id"       in data
        assert "chunks_retrieved" in data
        assert data["chunks_retrieved"] == 2

    def test_session_id_persists(self, client):
        with (
            patch("src.retrieval.retriever.Retriever.retrieve", return_value=[]),
            patch("src.llm.llm_client.AnthropicClient.chat", return_value="ok"),
        ):
            r1 = client.post("/api/chat", json={"query": "Q1", "provider": "anthropic"})
            sid = r1.json()["session_id"]
            r2  = client.post("/api/chat", json={"query": "Q2",
                                                  "provider": "anthropic",
                                                  "session_id": sid})
        assert r2.json()["session_id"] == sid

    def test_mission_filter_forwarded(self, client):
        with (
            patch("src.retrieval.retriever.Retriever.retrieve",
                  return_value=[]) as mock_retrieve,
            patch("src.llm.llm_client.AnthropicClient.chat", return_value="ok"),
        ):
            client.post("/api/chat", json={
                "query":          "PLATO docs",
                "provider":       "anthropic",
                "mission_filter": "PLATO",
            })
        _, kwargs = mock_retrieve.call_args
        assert kwargs.get("mission_name") == "PLATO"


class TestIngestEndpoint:

    def test_ingest_csv(self, client):
        csv_bytes = b"component,status\nMCS,operational\nMPS,planned\n"
        with (
            patch("src.embeddings.embedder.Embedder.embed_batch",
                  return_value=[[0.1]*1536, [0.2]*1536]),
            patch("src.vectordb.vector_store.VectorStore.upsert", return_value=2),
        ):
            resp = client.post(
                "/api/ingest",
                data={"mission_name": "PLATO"},
                files={"file": ("systems.csv", csv_bytes, "text/csv")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mission_name"]  == "PLATO"
        assert data["document_name"] == "systems.csv"
        assert data["chunks_created"] > 0

    def test_ingest_unsupported_format_returns_400(self, client):
        resp = client.post(
            "/api/ingest",
            data={"mission_name": "Gaia"},
            files={"file": ("doc.docx", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_ingest_quota_error_suggests_alternatives(self, client):
        with patch(
            "src.embeddings.embedder.Embedder.embed_batch",
            side_effect=Exception(
                "Error code: 429 - {'error': {'message': 'You exceeded your current quota', 'type': 'insufficient_quota'}}"
            ),
        ):
            resp = client.post(
                "/api/ingest",
                data={"mission_name": "PLATO"},
                files={"file": ("systems.csv", b"component,status\nMCS,operational\n", "text/csv")},
            )

        assert resp.status_code == 429
        detail = resp.json()["detail"]
        assert "Anthropic" in detail
        assert "Google Gemini" in detail
        assert "Ollama" in detail


class TestDeleteEndpoints:

    def test_delete_mission(self, client):
        with patch("src.vectordb.vector_store.VectorStore.delete_by_mission"):
            resp = client.delete("/api/missions/PLATO")
        assert resp.status_code == 200
        assert "PLATO" in resp.json()["message"]

    def test_delete_document(self, client):
        with patch("src.vectordb.vector_store.VectorStore.delete_by_document"):
            resp = client.delete("/api/documents/PLATO_ICD.pdf")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════
# Helpers tests
# ══════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_format_sources_deduplicates(self):
        from src.utils.helpers import format_sources

        chunks = [
            {"metadata": {"document_name": "ICD.pdf", "mission_name": "PLATO", "page": 1}},
            {"metadata": {"document_name": "ICD.pdf", "mission_name": "PLATO", "page": 1}},
            {"metadata": {"document_name": "SRS.pdf", "mission_name": "PLATO", "page": 5}},
        ]
        result = format_sources(chunks)
        assert result.count("ICD.pdf") == 1
        assert result.count("SRS.pdf") == 1

    def test_sanitise_filename(self):
        from src.utils.helpers import sanitise_filename

        assert sanitise_filename("PLATO ICD v1.2.pdf") == "PLATO ICD v1.2.pdf"
        assert "/" not in sanitise_filename("path/to/file.pdf")
        assert "\\" not in sanitise_filename("win\\path\\file.pdf")
