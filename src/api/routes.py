"""
FastAPI application factory and all API routes.

Endpoints:
    GET  /                          → Serve chat UI
    GET  /api/health                → Health + Pinecone stats
    GET  /api/stats                 → Pinecone index stats
    POST /api/chat                  → RAG chat turn
    POST /api/ingest                → Upload and ingest a document
    GET  /api/missions              → List distinct missions in the index
    DELETE /api/missions/{mission}  → Delete all documents for a mission
    DELETE /api/documents/{doc}     → Delete a specific document
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.chunking   import DocumentChunker
from src.embeddings import Embedder
from src.ingestion  import DocumentLoader
from src.llm        import LLMFactory
from src.prompts    import SYSTEM_PROMPT, build_no_context_response, build_rag_prompt
from src.retrieval  import Retriever
from src.vectordb   import VectorStore


# ── Pydantic models ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:          str
    provider:       str  = "anthropic"
    mission_filter: str | None = None
    session_id:     str | None = None


class SourceItem(BaseModel):
    document: str
    mission:  str
    page:     Any
    score:    float
    excerpt:  str


class ChatResponse(BaseModel):
    answer:           str
    sources:          list[SourceItem]
    provider:         str
    session_id:       str
    chunks_retrieved: int


class IngestResponse(BaseModel):
    document_name:    str
    mission_name:     str
    chunks_created:   int
    vectors_upserted: int


class HealthResponse(BaseModel):
    status:              str
    pinecone_connected:  bool
    total_vector_count:  int
    available_providers: list[str]


# ── Session store (swap for Redis in production) ───────────────────────────

_sessions: dict[str, list[dict[str, str]]] = {}
_MAX_HISTORY = 20   # messages kept per session


# ── App factory ────────────────────────────────────────────────────────────

def create_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(
        title=config["api"]["title"],
        version=config["api"]["version"],
        description=(
            "RAG-powered chat application for ESA ground segment documentation. "
            "Supports multi-mission document ingestion (PDF/CSV) and natural "
            "language querying via OpenAI, Anthropic, or Google LLMs."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Shared components ─────────────────────────────────────────────────
    embedder = Embedder(model=config["embeddings"]["model"])
    vector_store = VectorStore(
        index_name=config["pinecone"]["index_name"],
        dimension=embedder.dimension,
        metric=config["pinecone"]["metric"],
    )
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=config["retrieval"]["top_k"],
        score_threshold=config["retrieval"]["score_threshold"],
    )
    loader  = DocumentLoader()
    chunker = DocumentChunker(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )
    ns = config["pinecone"].get("namespace", "esa-missions")

    # ── Static UI ─────────────────────────────────────────────────────────
    if Path("static").exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")

    # ══════════════════════════════════════════════════════════════════════
    # Routes
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def serve_ui():
        ui = Path("static/index.html")
        if ui.exists():
            return FileResponse(ui)
        return HTMLResponse(
            "<h2>ESA Ground Segment RAG</h2>"
            "<p>Frontend not found. Place <code>static/index.html</code> "
            "or use the <a href='/docs'>API docs</a>.</p>"
        )

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    async def health():
        try:
            stats = vector_store.get_stats(ns)
            count = stats.get("total_vector_count", 0)
            ok    = True
        except Exception as exc:
            logger.warning(f"Pinecone health check failed: {exc}")
            count, ok = 0, False
        return HealthResponse(
            status="ok",
            pinecone_connected=ok,
            total_vector_count=count,
            available_providers=LLMFactory.available_providers(),
        )

    @app.get("/api/stats", tags=["System"])
    async def stats():
        return vector_store.get_stats(ns)

    # ── Chat ──────────────────────────────────────────────────────────────

    @app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(req: ChatRequest):
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # Session
        session_id = req.session_id or str(uuid.uuid4())
        history    = _sessions.get(session_id, [])

        # Retrieve context
        chunks = retriever.retrieve(
            query=req.query,
            mission_name=req.mission_filter,
            namespace=ns,
        )

        # Build user message
        if chunks:
            context      = retriever.format_context(chunks)
            history_text = "\n".join(
                f"{'Engineer' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in history[-6:]
            )
            user_msg = build_rag_prompt(req.query, context, history_text)
        else:
            user_msg = build_no_context_response(req.query, req.mission_filter)

        # LLM call
        try:
            provider_cfg = config["llm"]["providers"].get(req.provider, {})
            llm          = LLMFactory.create(req.provider, provider_cfg)
            answer       = llm.chat(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_msg,
                history=None,   # history is injected into the user message above
            )
        except Exception as exc:
            logger.error(f"LLM error [{req.provider}]: {exc}")
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

        # Persist session
        history.append({"role": "user",      "content": req.query})
        history.append({"role": "assistant", "content": answer})
        _sessions[session_id] = history[-_MAX_HISTORY:]

        # Format sources
        sources = [
            SourceItem(
                document=c["metadata"].get("document_name", "Unknown"),
                mission =c["metadata"].get("mission_name",  "Unknown"),
                page    =c["metadata"].get("page", c["metadata"].get("row", "—")),
                score   =round(c["score"], 3),
                excerpt =c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
            )
            for c in chunks
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            provider=req.provider,
            session_id=session_id,
            chunks_retrieved=len(chunks),
        )

    # ── Ingest ────────────────────────────────────────────────────────────

    @app.post("/api/ingest", response_model=IngestResponse, tags=["Documents"])
    async def ingest(
        file: UploadFile = File(..., description="PDF or CSV document"),
        mission_name: str = Form(..., description="ESA mission name, e.g. PLATO, Gaia"),
    ):
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided.")
        ext = Path(file.filename).suffix.lower()
        if ext not in {".pdf", ".csv"}:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

        logger.info(f"Ingesting: {file.filename} | mission={mission_name}")
        try:
            raw_bytes = await file.read()
            docs      = loader.load_from_bytes(raw_bytes, file.filename, mission_name)
            chunks    = chunker.chunk(docs)
            texts     = [c["text"] for c in chunks]
            embeddings = embedder.embed_batch(texts)
            upserted  = vector_store.upsert(chunks, embeddings, namespace=ns)
        except Exception as exc:
            logger.error(f"Ingestion failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

        logger.info(
            f"Ingested '{file.filename}': "
            f"{len(chunks)} chunks → {upserted} vectors"
        )
        return IngestResponse(
            document_name=file.filename,
            mission_name=mission_name,
            chunks_created=len(chunks),
            vectors_upserted=upserted,
        )

    # ── Delete ────────────────────────────────────────────────────────────

    @app.delete("/api/missions/{mission_name}", tags=["Documents"])
    async def delete_mission(mission_name: str):
        try:
            vector_store.delete_by_mission(mission_name, namespace=ns)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"message": f"Deleted all vectors for mission '{mission_name}'."}

    @app.delete("/api/documents/{document_name}", tags=["Documents"])
    async def delete_document(document_name: str):
        try:
            vector_store.delete_by_document(document_name, namespace=ns)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"message": f"Deleted all vectors for document '{document_name}'."}

    return app
