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
from pydantic import BaseModel, Field

from src.embeddings import Embedder
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


class DebugChatRequest(BaseModel):
    query:          str
    provider:       str = "anthropic"
    mission_filter: str | None = None
    session_id:     str | None = None
    include_answer: bool = True


class DebugChatResponse(BaseModel):
    answer:           str | None
    sources:          list[SourceItem]
    provider:         str
    session_id:       str
    chunks_retrieved: int
    raw_chunks:       int
    context:          str
    prompt:           str
    history:          list[dict[str, str]]


class EvaluationCaseRequest(BaseModel):
    query:              str
    expected_documents: list[str] = Field(default_factory=list)
    mission_filter:     str | None = None


class EvaluationRequest(BaseModel):
    cases:    list[EvaluationCaseRequest]
    namespace: str | None = None
    top_k:    int | None = None


class EvaluationCaseResult(BaseModel):
    query:              str
    expected_documents: list[str]
    retrieved_documents: list[str]
    hit:                bool
    precision_at_k:     float
    recall_at_k:        float
    reciprocal_rank:    float
    top_score:          float | None
    chunks_retrieved:   int


class EvaluationResponse(BaseModel):
    total_cases:          int
    hit_rate:             float
    mean_precision_at_k:  float
    mean_recall_at_k:     float
    mean_reciprocal_rank: float
    cases:                list[EvaluationCaseResult]


# ── Session store (swap for Redis in production) ───────────────────────────

_sessions: dict[str, list[dict[str, str]]] = {}
_MAX_HISTORY = 20   # messages kept per session


def _format_ingest_error(exc: Exception) -> tuple[int, str]:
    message = str(exc)
    if "insufficient_quota" in message or "Error code: 429" in message:
        return (
            429,
            "OpenAI embedding quota was exceeded during ingestion. "
            "This app currently uses OpenAI for document embeddings, so chat provider "
            "switches like Anthropic, Google Gemini, or Ollama do not affect uploads. "
            "To avoid this error, add OpenAI billing/quota or switch the embeddings "
            "backend to a local provider such as Ollama embeddings and re-index the data.",
        )
    return 500, message


def _is_index_count_query(query: str) -> bool:
    lowered = query.lower()
    count_markers = (
        "how many",
        "number of",
        "count",
        "total",
    )
    corpus_markers = (
        "doc",
        "document",
        "documents",
        "docs",
        "vector",
        "vectors",
        "chunk",
        "chunks",
        "file",
        "files",
        "database",
        "index",
    )
    return any(marker in lowered for marker in count_markers) and any(
        marker in lowered for marker in corpus_markers
    )


def _format_history(history: list[dict[str, str]], limit: int = 6) -> str:
    return "\n".join(
        f"{'Engineer' if message['role']=='user' else 'Assistant'}: {message['content']}"
        for message in history[-limit:]
    )


def _history_snapshot(history: list[dict[str, str]], limit: int = 6) -> list[dict[str, str]]:
    return history[-limit:]


def _chunk_to_source_item(chunk: dict[str, Any]) -> SourceItem:
    return SourceItem(
        document=chunk["metadata"].get("document_name", "Unknown"),
        mission=chunk["metadata"].get("mission_name", "Unknown"),
        page=chunk["metadata"].get("page", chunk["metadata"].get("row", "—")),
        score=round(chunk["score"], 3),
        excerpt=chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""),
    )


def _normalize_document_name(name: str) -> str:
    return Path(name).name.strip().lower()


def _unique_retrieved_documents(chunks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    documents: list[str] = []
    for chunk in chunks:
        document_name = chunk["metadata"].get("document_name", "Unknown")
        normalized = _normalize_document_name(document_name)
        if normalized in seen:
            continue
        seen.add(normalized)
        documents.append(document_name)
    return documents


def _evaluate_case(
    retriever: Retriever,
    query: str,
    expected_documents: list[str],
    mission_filter: str | None,
    namespace: str,
) -> EvaluationCaseResult:
    inspection = retriever.inspect_retrieval(
        query=query,
        mission_name=mission_filter,
        namespace=namespace,
    )
    chunks = inspection["chunks"]
    retrieved_documents = _unique_retrieved_documents(chunks)
    expected_normalized = {
        _normalize_document_name(document)
        for document in expected_documents
        if document.strip()
    }
    retrieved_normalized = [
        _normalize_document_name(document)
        for document in retrieved_documents
    ]
    matched_documents = expected_normalized.intersection(retrieved_normalized)
    hit = bool(matched_documents)

    first_match_rank = next(
        (index + 1 for index, document in enumerate(retrieved_normalized) if document in expected_normalized),
        None,
    )
    reciprocal_rank = 1.0 / first_match_rank if first_match_rank else 0.0

    precision_at_k = (
        len(matched_documents) / len(retrieved_documents)
        if retrieved_documents
        else 0.0
    )
    recall_at_k = (
        len(matched_documents) / len(expected_normalized)
        if expected_normalized
        else 0.0
    )

    top_score = chunks[0]["score"] if chunks else None
    return EvaluationCaseResult(
        query=query,
        expected_documents=expected_documents,
        retrieved_documents=retrieved_documents,
        hit=hit,
        precision_at_k=round(precision_at_k, 3),
        recall_at_k=round(recall_at_k, 3),
        reciprocal_rank=round(reciprocal_rank, 3),
        top_score=round(top_score, 3) if top_score is not None else None,
        chunks_retrieved=len(chunks),
    )


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
    embedder = None
    vector_store = None
    retriever = None
    loader = None
    chunker = None

    def _get_embedder():
        nonlocal embedder
        if embedder is None:
            embedder = Embedder(
                provider=config["embeddings"].get("provider", "openai"),
                model=config["embeddings"]["model"],
            )
        return embedder

    def _get_vector_store():
        nonlocal vector_store
        if vector_store is None:
            current_embedder = _get_embedder()
            vector_store = VectorStore(
                index_name=config["pinecone"]["index_name"],
                dimension=current_embedder.dimension,
                metric=config["pinecone"]["metric"],
            )
        return vector_store

    def _get_retriever():
        nonlocal retriever
        if retriever is None:
            retriever = Retriever(
                embedder=_get_embedder(),
                vector_store=_get_vector_store(),
                top_k=config["retrieval"]["top_k"],
                score_threshold=config["retrieval"]["score_threshold"],
            )
        return retriever

    def _get_loader():
        nonlocal loader
        if loader is None:
            from src.ingestion import DocumentLoader

            loader = DocumentLoader()
        return loader

    def _get_chunker():
        nonlocal chunker
        if chunker is None:
            from src.chunking import DocumentChunker

            chunker = DocumentChunker(
                chunk_size=config["chunking"]["chunk_size"],
                chunk_overlap=config["chunking"]["chunk_overlap"],
            )
        return chunker
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
            return HTMLResponse(
                ui.read_text(encoding="utf-8"),
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse(
            "<h2>ESA Ground Segment Documentation RAG</h2>"
            "<p>Frontend not found. Place <code>static/index.html</code> "
            "or use the <a href='/docs'>API docs</a>.</p>"
        )

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    async def health():
        try:
            stats = _get_vector_store().get_stats(ns)
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
        return _get_vector_store().get_stats(ns)

    # ── Chat ──────────────────────────────────────────────────────────────

    @app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(req: ChatRequest):
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # Session
        session_id = req.session_id or str(uuid.uuid4())
        history    = _sessions.get(session_id, [])

        if _is_index_count_query(req.query):
            stats = _get_vector_store().get_stats(ns)
            namespace_stats = stats.get("namespaces", {}).get(ns, {})
            count = namespace_stats.get("vector_count", stats.get("total_vector_count", 0))
            answer = (
                f"There are {count} indexed chunks in the current database namespace. "
                "Each chunk comes from an ingested document page or row."
            )

            history.append({"role": "user", "content": req.query})
            history.append({"role": "assistant", "content": answer})
            _sessions[session_id] = history[-_MAX_HISTORY:]

            return ChatResponse(
                answer=answer,
                sources=[],
                provider=req.provider,
                session_id=session_id,
                chunks_retrieved=0,
            )

        # Retrieve context
        chunks = _get_retriever().retrieve(
            query=req.query,
            mission_name=req.mission_filter,
            namespace=ns,
        )

        # Build user message
        if chunks:
            context      = _get_retriever().format_context(chunks)
            history_text = _format_history(history)
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

    @app.post("/api/debug/chat", response_model=DebugChatResponse, tags=["Debug"])
    async def debug_chat(req: DebugChatRequest):
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        session_id = req.session_id or str(uuid.uuid4())
        history = _sessions.get(session_id, [])
        inspection = _get_retriever().inspect_retrieval(
            query=req.query,
            mission_name=req.mission_filter,
            namespace=ns,
        )
        chunks = inspection["chunks"]
        context = _get_retriever().format_context(chunks)
        history_text = _format_history(history)
        prompt = (
            build_rag_prompt(req.query, context, history_text)
            if chunks
            else build_no_context_response(req.query, req.mission_filter)
        )

        answer: str | None = None
        if req.include_answer:
            try:
                provider_cfg = config["llm"]["providers"].get(req.provider, {})
                llm = LLMFactory.create(req.provider, provider_cfg)
                answer = llm.chat(
                    system_prompt=SYSTEM_PROMPT,
                    user_message=prompt,
                    history=None,
                )
            except Exception as exc:
                logger.error(f"LLM error [{req.provider}]: {exc}")
                raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

            history.append({"role": "user", "content": req.query})
            history.append({"role": "assistant", "content": answer})
            _sessions[session_id] = history[-_MAX_HISTORY:]

        return DebugChatResponse(
            answer=answer,
            sources=[_chunk_to_source_item(chunk) for chunk in chunks],
            provider=req.provider,
            session_id=session_id,
            chunks_retrieved=len(chunks),
            raw_chunks=inspection["raw_count"],
            context=context,
            prompt=prompt,
            history=_history_snapshot(history),
        )

    @app.post("/api/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
    async def evaluate(req: EvaluationRequest):
        if not req.cases:
            raise HTTPException(status_code=400, detail="At least one evaluation case is required.")

        retriever = _get_retriever()
        namespace = req.namespace or ns
        original_top_k = retriever.top_k
        if req.top_k is not None:
            retriever.top_k = req.top_k

        try:
            results = [
                _evaluate_case(
                    retriever=retriever,
                    query=case.query,
                    expected_documents=case.expected_documents,
                    mission_filter=case.mission_filter,
                    namespace=namespace,
                )
                for case in req.cases
            ]
        finally:
            retriever.top_k = original_top_k

        total_cases = len(results)
        hit_count = sum(1 for result in results if result.hit)
        return EvaluationResponse(
            total_cases=total_cases,
            hit_rate=round(hit_count / total_cases, 3),
            mean_precision_at_k=round(
                sum(result.precision_at_k for result in results) / total_cases,
                3,
            ),
            mean_recall_at_k=round(
                sum(result.recall_at_k for result in results) / total_cases,
                3,
            ),
            mean_reciprocal_rank=round(
                sum(result.reciprocal_rank for result in results) / total_cases,
                3,
            ),
            cases=results,
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
            docs      = _get_loader().load_from_bytes(raw_bytes, file.filename, mission_name)
            chunks    = _get_chunker().chunk(docs)
            texts     = [c["text"] for c in chunks]
            embeddings = _get_embedder().embed_batch(texts)
            upserted  = _get_vector_store().upsert(chunks, embeddings, namespace=ns)
        except Exception as exc:
            logger.error(f"Ingestion failed: {exc}")
            status_code, detail = _format_ingest_error(exc)
            raise HTTPException(status_code=status_code, detail=detail)

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
            _get_vector_store().delete_by_mission(mission_name, namespace=ns)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"message": f"Deleted all vectors for mission '{mission_name}'."}

    @app.delete("/api/documents/{document_name}", tags=["Documents"])
    async def delete_document(document_name: str):
        try:
            _get_vector_store().delete_by_document(document_name, namespace=ns)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"message": f"Deleted all vectors for document '{document_name}'."}

    return app
