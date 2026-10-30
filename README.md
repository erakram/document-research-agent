# Document Research Agent

![tests](https://github.com/erakram/document-research-agent/actions/workflows/tests.yml/badge.svg)

**▶ Live demo: [document-research-agent.onrender.com](https://document-research-agent.onrender.com/)**
*(free hosting: the first visit after a quiet period takes about a minute to wake up)*

Upload PDFs or text files and ask questions about them. An AI agent searches the documents, can search again with different wording, and answers **with citations to the exact passages** it used. If the answer isn't in the documents, it says so instead of making something up.

Built with **Groq** (LLM function calling), **model2vec embeddings**, **SQLite** (vector storage + FTS5 full-text search), and **FastAPI**.

<!-- Add a screenshot or demo GIF here: ![demo](docs/demo.gif) -->

## How it works

```
Upload ─► extract text per page ─► split into overlapping chunks ─► embed ─► SQLite
                                                                   (vectors + FTS5 index)

Question ─► Agent loop ──► Groq LLM ──► search_documents(query)
               ▲                              │
               │        hybrid search: cosine similarity (embeddings)
               │                     + BM25 keyword match (FTS5)
               │                     merged with Reciprocal Rank Fusion
               └──── passages with source ids [S12] ◄──┘
                                   │
Answer with [S12] citations ─► keep only citations that were really retrieved ─► UI
```

**Ingestion.** PDFs are read page by page with `pypdf`. The text is split into ~150-word chunks that overlap by 30 words, so a fact split across a boundary is still found. Each chunk keeps its document name and page number.

**Hybrid retrieval.** Embeddings find passages with the same *meaning* ("work from home" ↔ "work remotely"). Keyword search finds *exact terms* that embeddings can miss (product names, codes like "AES-256"). The two rankings are merged with Reciprocal Rank Fusion.

**Agentic RAG.** Instead of one fixed search, the LLM decides what to search for. It can rephrase, or split a multi-part question into several searches.

**Grounded answers.** Each passage has a source id like `[S12]`. The answer must cite them. The backend keeps only citations that point to passages the agent actually retrieved, so the UI never shows a made-up source. Clicking a citation shows the original passage, document and page.

**Missing information.** The prompt requires answers to come only from the documents. The agent has a step budget, and on the final step tools are switched off, so it must answer from what it found or say "I couldn't find this in the documents" instead of searching forever.

## Engineering details

- **Hybrid search** — embeddings + SQLite FTS5 (BM25), merged with RRF. User input is quoted, so it can't inject FTS5 query syntax.
- **Free embeddings** — `model2vec` runs on CPU with only NumPy: no GPU and no paid embeddings API. It loads in about 1 s and fits on a free hosting plan. The `Embedder` interface makes it easy to swap in a hosted model.
- **Vector search** — vectors are stored as BLOBs and searched with one NumPy matrix multiply, with the matrix cached in memory. Fine for thousands of chunks. For millions you would swap in pgvector or Qdrant behind the same `search()` method.
- **Token efficiency** — passages the LLM has already seen aren't resent. Reasoning models run at low reasoning effort, which keeps latency and free-tier usage down.
- **Reliability** — timeouts, automatic retries with backoff on rate limits and 5xx errors, a step limit, and handling for model quirks (for example, normalizing `【S1】` citations to `[S1]`).
- **Safe uploads** — type and size limits, path components stripped from file names, and clear errors for scanned PDFs with no text.
- **Observability** — request IDs, plus latency for each request, LLM step and search.
- **Tests** — 36 pytest tests covering ingestion, chunking, hybrid search, citations, the agent loop (with a fake LLM and a fake embedder, so no API key is needed) and the API.

## Run it locally

Requires Python 3.11+ and a free Groq API key from <https://console.groq.com/keys>.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your key into .env
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. Two sample documents (an employee handbook and a product guide) load on first start. Upload your own PDFs from the sidebar.

Run the tests:

```bash
pytest -q
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload a file (multipart `file`) |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Delete a document and its chunks |
| `POST` | `/api/ask` | Ask a question |
| `GET` | `/api/health` | Status, document count, whether the LLM is configured |

Example `POST /api/ask`:

```json
{ "question": "How much is the meal allowance when travelling?", "history": [] }
```

```json
{
  "answer": "Meals during business travel are covered up to USD 60 per day [S8].",
  "citations": [
    { "source_id": "S8", "document": "northwind-employee-handbook.md", "page": null,
      "text": "... During business travel, meals are covered up to USD 60 per day ..." }
  ],
  "tool_calls": [{ "name": "search_documents", "arguments": { "query": "meal allowance travel" } }],
  "latency_ms": 900
}
```

Interactive API docs are at `/docs`.

## Deploy (free)

1. Push this folder to a GitHub repository.
2. On [Render](https://render.com), choose **New → Blueprint** and select the repo (it uses `render.yaml` and the `Dockerfile`, which downloads the embedding model at build time).
3. Enter your `GROQ_API_KEY` when prompted.

The free plan's disk is temporary, so uploaded files are cleared when the service restarts, and the sample documents reload automatically. On Groq's free tier, rapid back-to-back questions can hit the rate limit; the app retries automatically.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required for questions |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any Groq model with tool calling |
| `EMBEDDING_MODEL` | `minishlab/potion-base-8M` | Any model2vec model |
| `MAX_AGENT_STEPS` | `5` | Max LLM calls per question |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `REQUEST_TIMEOUT` / `MAX_RETRIES` | `30` / `3` | LLM request timeout and retries |
| `SEED_SAMPLE_DOCS` | `true` | Load `sample_docs/` when the store is empty |

## Project structure

```
app/
  main.py        FastAPI app: upload, ask, health, error handling
  agent.py       Agent loop, prompt, citation checking
  store.py       SQLite store: vectors, FTS5, hybrid search
  ingest.py      PDF/text extraction and chunking
  embeddings.py  model2vec wrapper behind an Embedder interface
  config.py      Settings from environment variables
  static/        Web UI (plain HTML/JS)
sample_docs/     Fictional sample documents
tests/           pytest suite (no API key or model download needed)
```
