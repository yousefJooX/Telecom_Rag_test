# Telecom RAG API

A production-style RAG (Retrieval Augmented Generation) backend for telecom customer support, built with FastAPI, LangChain, and FAISS.

> **Status: v1 — foundation release.** This version ships the project skeleton (config, logging, model factory, FastAPI app) with a working `/health` endpoint. Ingestion, retrieval, and the query endpoints are being built next — see [Roadmap](#roadmap).

---

## Why this structure

Most RAG demos live in a single Jupyter notebook — great for prototyping, unusable in production. This project follows a layered **Controller → Service → Repository** architecture instead, so each concern (HTTP handling, business logic, data access) lives in its own place and can be changed, tested, or swapped independently.

## Features

- **Multi-provider LLM support** — switch between **Gemini**, **OpenAI**, and **NVIDIA NIM** by changing one line in `config.yml`, no code changes required (Factory Pattern via `ModelFactory`)
- **Config-driven** — every tunable value (chunk size, model names, retrieval `k`) lives in `config/config.yml`, not hardcoded in Python
- **GPU-accelerated embeddings** — HuggingFace sentence-transformers run on CUDA when available
- **Structured logging** — console + rotating file handler (`app.log`), shared across every layer
- **Singleton model loading** — embedding and LLM clients are built once (`@lru_cache`) and reused across every request, not reloaded per call

## Project structure

```
Telecom_Rag_test/
├── config/
│   └── config.yml            # app, data, and model settings
├── src/
│   ├── config/
│   │   └── config_parser.py  # loads config.yml + .env into a typed Config object
│   ├── logging/
│   │   └── logger.py         # shared logger (console + rotating file)
│   └── core/
│       └── factories.py      # ModelFactory — builds embeddings + LLM (Gemini/OpenAI/NVIDIA)
├── .env.example               # template for required API keys
├── .gitignore
├── main.py                    # FastAPI app entrypoint + startup warmup
└── requirements.txt
```

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/yousefJooX/Telecom_Rag_test.git
cd Telecom_Rag_test
git checkout dev

uv venv
source .venv/bin/activate.fish   # fish shell — use .venv/bin/activate for bash/zsh
uv pip install -r requirements.txt
```

### 2. Configure your API keys

```bash
cp .env.example .env
```

Edit `.env` and add the key for whichever provider you're using (only one is required, matching `llm_provider` in `config.yml`):

```env
GOOGLE_API_KEY="your_gemini_key"
OPENAI_API_KEY="your_openai_key"
NVIDIA_API_KEY="your_nvidia_key"
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

Visit `http://localhost:8000/health` to confirm it's running, and `http://localhost:8000/docs` for the interactive API docs.

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

## Requirements

- Python 3.10+
- (Optional) NVIDIA GPU + CUDA-enabled PyTorch for faster embedding generation

## Roadmap

- [ ] `/api/v1/ingest` — upload and index documents into FAISS
- [ ] `/api/v1/query` — retrieve context + generate LLM response
- [ ] `VectorDatabaseRepository` — FAISS repository layer with in-memory caching
- [ ] `IngestionService` / `RAGService` — business logic layer
- [ ] Unit + integration tests
- [ ] Dockerfile for containerized deployment

## License

TBD
