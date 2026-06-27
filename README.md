# ESA Ground Segment RAG Assistant

Natural language chat interface for querying ESA spacecraft ground segment documentation.  
Upload PDFs and CSVs, then ask questions in plain English — answers include source citations and relevance scores.

```
┌─────────────┐    PDF/CSV     ┌───────────┐   embed    ┌──────────┐
│  Documents  │ ─────────────► │  Ingest   │ ─────────► │ Pinecone │
└─────────────┘                │  Pipeline │            │  Index   │
                               └───────────┘            └──────────┘
                                                              │
┌─────────────┐    question    ┌───────────┐   retrieve       │
│  Engineer   │ ─────────────► │    RAG    │ ◄───────────────┘
│  (browser)  │ ◄───────────── │  Pipeline │
└─────────────┘    answer+     └─────┬─────┘
                   citations         │ prompt + context
                               ┌─────▼─────┐
                               │    LLM    │  OpenAI / Anthropic / Google
                               └───────────┘
```

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo>
cd rag_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   OPENAI_API_KEY    — for embeddings (always required) + GPT-4o
#   ANTHROPIC_API_KEY — for Claude
#   GOOGLE_API_KEY    — for Gemini
#   PINECONE_API_KEY  — for the vector store
```

### 3. Run

```bash
python main.py
# → http://localhost:8000
```

Open your browser at `http://localhost:8000` to access the chat UI.  
Interactive API docs are at `http://localhost:8000/docs`.

---

## Project structure

```
rag_project/
├── main.py                  # Entry point — loads config, starts uvicorn
├── config.yaml              # All tunable parameters
├── requirements.txt
├── .env.example             # Copy to .env and fill in API keys
├── .gitignore
│
├── src/
│   ├── ingestion/
│   │   └── loader.py        # Load PDF pages and CSV rows with metadata
│   ├── chunking/
│   │   └── chunker.py       # RecursiveCharacterTextSplitter, deterministic IDs
│   ├── embeddings/
│   │   └── embedder.py      # OpenAI text-embedding-3-small, batch support
│   ├── vectordb/
│   │   └── vector_store.py  # Pinecone: upsert, query, delete, stats
│   ├── retrieval/
│   │   └── retriever.py     # Embed query → Pinecone search → format context
│   ├── prompts/
│   │   └── prompt_templates.py  # System prompt, RAG template, no-context fallback
│   ├── llm/
│   │   └── llm_client.py    # OpenAIClient, AnthropicClient, GoogleClient + factory
│   ├── api/
│   │   └── routes.py        # FastAPI app factory + all endpoints
│   └── utils/
│       └── helpers.py       # Config loader, logging setup, formatting helpers
│
├── static/
│   └── index.html           # Dark-theme single-page chat UI
│
├── tests/
│   └── test_app.py          # Unit + integration tests (pytest)
│
└── logs/
    └── app.log              # Rotating log (auto-created)
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serve chat UI |
| `GET`  | `/api/health` | Pinecone connection + vector count |
| `GET`  | `/api/stats` | Full index statistics |
| `POST` | `/api/chat` | RAG chat turn |
| `POST` | `/api/ingest` | Upload and index a PDF or CSV |
| `DELETE` | `/api/missions/{name}` | Delete all vectors for a mission |
| `DELETE` | `/api/documents/{name}` | Delete all vectors from a document |

### POST /api/chat

```json
{
  "query":          "What is the PLATO uplink architecture?",
  "provider":       "anthropic",
  "mission_filter": "PLATO",
  "session_id":     "optional-uuid-for-conversation-continuity"
}
```

Response:

```json
{
  "answer":           "The PLATO uplink subsystem uses …",
  "sources":          [{"document": "PLATO_MCS_ICD.pdf", "mission": "PLATO",
                        "page": 12, "score": 0.87, "excerpt": "…"}],
  "provider":         "anthropic",
  "session_id":       "abc-123",
  "chunks_retrieved": 3
}
```

### POST /api/ingest

Multipart form:
- `file` — PDF or CSV binary
- `mission_name` — e.g. `PLATO`, `Gaia`, `CHEOPS`

---

## Configuration (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `pinecone.index_name` | `esa-ground-segment` | Pinecone index to use |
| `pinecone.namespace` | `esa-missions` | Logical partition within the index |
| `embeddings.model` | `text-embedding-3-small` | OpenAI embedding model |
| `chunking.chunk_size` | `1000` | Max characters per chunk |
| `chunking.chunk_overlap` | `200` | Overlap between adjacent chunks |
| `retrieval.top_k` | `5` | Chunks retrieved per query |
| `retrieval.score_threshold` | `0.65` | Minimum cosine similarity to return |
| `llm.default_provider` | `anthropic` | Default LLM provider |

---

## Adding a new LLM provider

1. Create a new class in `src/llm/llm_client.py` extending `BaseLLMClient`
2. Implement `provider_name` and `chat()`
3. Register it in `LLMFactory._REGISTRY` and `_DEFAULTS`
4. Add the provider key under `llm.providers` in `config.yaml`

---

## Running tests

```bash
pytest tests/ -v
```

External services (Pinecone, OpenAI, Anthropic, Google) are mocked in all tests.  
No API keys are required to run the test suite.

---

## Supported document types

| Format | How it's loaded | Metadata captured |
|--------|-----------------|-------------------|
| PDF | `pypdf` — one dict per non-empty page | `page`, `total_pages` |
| CSV | `pandas` — one dict per row | `row`, `columns` |

---

## Production notes

- **Session store** — conversation history is held in-memory (`dict`). For multi-worker or persistent deployments replace `_sessions` in `routes.py` with a Redis-backed store.
- **Embeddings** — only OpenAI is wired for embeddings. The model and dimension are configured in `config.yaml`; changing model requires re-indexing all documents.
- **Pinecone tier** — the default `ServerlessSpec` targets AWS `us-east-1`. Adjust in `vector_store.py` to match your Pinecone plan/region.
- **Auth** — no authentication is implemented. Add FastAPI `Depends` middleware or an API-key header before exposing to a network.
