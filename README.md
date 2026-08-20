# Telecom RAG API

A production-style RAG (Retrieval Augmented Generation) backend for telecom customer support, built with FastAPI, LangChain, and FAISS.

---

## Why this structure

Most RAG demos live in a single Jupyter notebook — great for prototyping, unusable in production. This project follows a layered **Controller → Service → Repository** architecture instead, so each concern (HTTP handling, business logic, data access) lives in its own place and can be changed, tested, or swapped independently.

## Features

- **Multi-provider LLM support** — switch between **Gemini**, **OpenAI**, and **NVIDIA NIM** by changing one line in `config.yml`, no code changes required (Factory Pattern via `ModelFactory`)
- **Config-driven** — every tunable value (chunk size, model names, retrieval `k`, token pricing) lives in `config/config.yml`, not hardcoded in Python
- **GPU/CPU auto-detection** — embeddings run on CUDA when available, fall back to CPU automatically
- **Structured logging** — console + rotating file handler (`app.log`), shared across every layer
- **Singleton model loading** — embedding and LLM clients are built once (`@lru_cache`) and reused across every request, not reloaded per call
- **Token & cost tracking** — per-query token usage and cost are logged and aggregated, exposed via `/api/v1/stats`
- **In-memory FAISS caching** — the vector index is loaded into RAM once (on startup or first query), zero disk I/O on subsequent requests
- **Dockerized** — `Dockerfile` + `docker-compose.yml` with named volume for persistent FAISS index across container rebuilds
- **CI/CD** — GitHub Actions workflow runs the full test suite on every push/PR to `dev` and `main`

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — returns app name, version, and docs URL |
| `POST` | `/api/v1/ingest` | Upload and index a `.txt`, `.md`, or `.csv` file into FAISS |
| `POST` | `/api/v1/query` | Submit a customer ticket, get a RAG-generated response with token usage |
| `GET` | `/api/v1/stats` | View aggregated token usage and cost across all queries |

Interactive docs available at `/docs` (Swagger UI) once the server is running.

## Project structure

```
Telecom_Rag_test/
├── config/
│   └── config.yml                 # app, data, and model settings (single source of truth)
├── src/
│   ├── config/
│   │   └── config_parser.py       # loads config.yml + .env into a typed Config object
│   ├── logging/
│   │   └── logger.py             # shared logger (console + rotating file handler)
│   ├── core/
│   │   └── factories.py           # ModelFactory — Factory + Singleton for embeddings & LLM
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── routers/
│   │   ├── ingest_router.py        # POST /api/v1/ingest (Controller layer)
│   │   └── query_router.py       # POST /api/v1/query, GET /api/v1/stats (Controller layer)
│   ├── services/
│   │   ├── ingest_service.py       # IngestionService — file processing & chunking (Service layer)
│   │   └── rag_service.py        # RAGService — retrieval + generation + cost tracking (Service layer)
│   └── vectorstore/
│       └── database.py           # VectorDatabaseRepository — FAISS operations (Repository layer)
├── tests/
│   ├── test_health.py             # Health endpoint test
│   ├── test_ingest.py             # Ingest endpoint tests (extension validation, .txt, .md)
│   └── test_query.py             # Query endpoint tests (empty ticket 422, mocked LLM success)
├── .env.example                   # template for required API keys
├── .dockerignore
├── .github/workflows/ci.yml       # CI — runs pytest on push/PR
├── Dockerfile
├── docker-compose.yml              # named volume for persistent FAISS index
├── main.py                        # FastAPI app entrypoint + lifespan startup
└── requirements.txt
```

## Architecture

```
HTTP Request
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│   Routers    │ ──▶ │     Services     │ ──▶ │      Repository        │
│ (Controllers)│     │ (Business Logic)  │     │  (FAISS Data Access)   │
└──────────────┘     └──────────────────┘     └─────────────────────────┘
    │                        │                          │
    │ FastAPI + Pydantic    │ LangChain chains           │ FAISS index (in-memory)
    │ HTTP concerns only     │ RAG orchestration         │ Only layer touching vectors
    │                       │ Token/cost tracking      │ Disk I/O only on first load
```

**Design patterns:**
- **Factory** (`ModelFactory`) — switches LLM/embedding providers via config, no code changes
- **Singleton** (`@lru_cache(maxsize=1)`) — models loaded once, reused across requests
- **Repository** (`VectorDatabaseRepository`) — encapsulates all FAISS operations behind a clean interface

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/yousefJooX/Telecom_Rag_test.git
cd Telecom_Rag_test
git checkout dev

uv venv
source .venv/bin/activate   # or .venv/bin/activate.fish for fish shell
uv pip install -r requirements.txt
```

### 2. Configure your API keys

```bash
cp .env.example .env
```

Edit `.env` and add the key for whichever provider you're using (only one is required, matching `llm_provider` in `config.yml`):

```env
GOOGLE_API_KEY="AIzaSy...your_real_key_here"
OPENAI_API_KEY="sk-...your_real_key"
NVIDIA_API_KEY="nvapi-...your_real_key"
```

### 3. Choose your model provider

Edit `config/config.yml`:

```yaml
models:
  llm_provider: "gemini"        # "gemini" | "openai" | "nvidia"
  llm_model_name: "gemini-2.5-flash"
```

### 4. Run the server

```bash
python3 main.py
```

Visit `http://localhost:8000/` for the health check, and `http://localhost:8000/docs` for the interactive API docs.

### 5. Ingest a document and query it

```bash
# Upload a knowledge base file
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@data/Telecom_Internal_KB.txt"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"ticket": "My internet keeps dropping every hour"}'

# Check aggregated token usage and cost
curl http://localhost:8000/api/v1/stats
```

## Docker

```bash
docker compose up --build
```

The FAISS index is persisted via a named Docker volume (`faiss_telecom_index`), so it survives container rebuilds. The index is loaded into memory on startup if it exists; otherwise, ingest a file to create it.

## Configuration reference

| Key | Description |
|---|---|
| `app.name`, `app.version` | Displayed in health check and API docs |
| `data.vector_index_path` | Local path where the FAISS index is saved/loaded |
| `data.chunk_size` / `chunk_overlap` | Text splitting parameters for ingestion |
| `data.batch_size` | Number of chunks embedded per batch during indexing |
| `models.embedding_provider` | Currently supports `huggingface` |
| `models.embedding_model_name` | Any sentence-transformers model name |
| `models.llm_provider` | `gemini` \| `openai` \| `nvidia` |
| `models.llm_model_name` | Model name matching the selected provider |
| `models.temperature` | LLM sampling temperature |
| `models.k_retrieval` | Number of chunks retrieved per query |
| `models.cost_per_1m_input_tokens` | Price per 1M input tokens (USD) — used for cost tracking |
| `models.cost_per_1m_output_tokens` | Price per 1M output tokens (USD) — used for cost tracking |

## Testing

```bash
pytest tests/ -v
```

| Test | What it covers |
|---|---|
| `test_health_endpoint` | Health check returns 200 with correct app metadata |
| `test_ingest_wrong_extension` | `.exe` upload returns 400 with error message |
| `test_ingest_valid_txt` | `.txt` upload returns 201 with chunk count |
| `test_ingest_valid_md` | `.md` upload returns 201 with chunk count |
| `test_query_empty_ticket` | Empty string ticket returns 422 (Pydantic validation) |
| `test_query_mocked_success` | Mocked RAGService returns full response with token tracking |

## Requirements

- Python 3.10+
- (Optional) NVIDIA GPU + CUDA for faster embedding generation (auto-detected)

## Roadmap

- [ ] RAG evaluation pipeline (retrieval quality metrics, golden-set benchmarking)
- [ ] Persistent token/cost storage (currently in-memory, lost on restart)
- [ ] Vector index refresh without full re-ingest (incremental updates)
- [ ] Authentication & rate limiting

## License

TBD
