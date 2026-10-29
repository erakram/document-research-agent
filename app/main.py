"""FastAPI app: upload/list/delete documents, ask questions, and serve the web page."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import groq
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import AgentStepLimitError, build_agent
from app.config import PROJECT_ROOT, Settings
from app.embeddings import Model2VecEmbedder
from app.ingest import SUPPORTED_TYPES, IngestError, chunk_pages, extract_pages
from app.store import DocumentStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

STATIC_DIR = PROJECT_ROOT / "app" / "static"
SAMPLE_DIR = PROJECT_ROOT / "sample_docs"


def ingest_file(store: DocumentStore, filename: str, data: bytes) -> dict:
    pages = extract_pages(filename, data)
    page_count = pages[-1].number if pages[0].number is not None else None
    return store.add_document(filename, chunk_pages(pages), pages=page_count)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    store = getattr(app.state, "store", None) or DocumentStore(
        settings.db_path, Model2VecEmbedder(settings.embedding_model))
    if settings.seed_sample_docs and not store.list_documents():
        for path in sorted(SAMPLE_DIR.glob("*")):
            if path.suffix.lower() in SUPPORTED_TYPES:
                ingest_file(store, path.name, path.read_bytes())
                logger.info("seeded sample document %s", path.name)
    app.state.settings = settings
    app.state.store = store
    app.state.agent = build_agent(settings, store) if settings.groq_api_key else None
    if app.state.agent is None:
        logger.warning("GROQ_API_KEY is not set; /api/ask will return 503.")
    yield


app = FastAPI(title="Document Research Agent", lifespan=lifespan)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class CitationOut(BaseModel):
    source_id: str
    document: str
    page: int | None
    text: str


class ToolCall(BaseModel):
    name: str
    arguments: dict


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    tool_calls: list[ToolCall]
    latency_ms: int


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    logger.info("request id=%s %s %s status=%d ms=%.0f", request_id, request.method,
                request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


@app.get("/api/documents")
def list_documents(request: Request) -> list[dict]:
    return request.app.state.store.list_documents()


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile, request: Request) -> dict:
    max_bytes = int(request.app.state.settings.max_upload_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File is larger than {request.app.state.settings.max_upload_mb:g} MB.")
    filename = Path(file.filename or "upload.txt").name  # drop any path the client sent
    try:
        # Parsing and embedding are CPU work, so keep them off the event loop.
        return await run_in_threadpool(ingest_file, request.app.state.store, filename, data)
    except IngestError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: int, request: Request) -> None:
    if not request.app.state.store.delete_document(document_id):
        raise HTTPException(404, "Document not found.")


@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(503, "The server has no GROQ_API_KEY configured.")
    if not request.app.state.store.list_documents():
        raise HTTPException(400, "Upload a document first.")

    start = time.perf_counter()
    try:
        reply = await run_in_threadpool(agent.ask, body.question, [t.model_dump() for t in body.history])
    except groq.RateLimitError:
        raise HTTPException(429, "The AI service is busy. Please try again in a minute.")
    except groq.APITimeoutError:
        raise HTTPException(504, "The AI service took too long to respond. Please try again.")
    except groq.APIConnectionError:
        raise HTTPException(503, "Could not reach the AI service. Please try again.")
    except groq.AuthenticationError:
        logger.error("Groq rejected the API key.")
        raise HTTPException(503, "The server's AI credentials are invalid.")
    except AgentStepLimitError:
        raise HTTPException(502, "I couldn't work that out. Try rephrasing your question.")
    except groq.APIStatusError as e:
        logger.error("Groq error status=%s: %s", e.status_code, e.message)
        raise HTTPException(502, "The AI service returned an error. Please try again.")

    return AskResponse(
        answer=reply.answer,
        citations=[CitationOut(**vars(c)) for c in reply.citations],
        tool_calls=reply.tool_calls,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )


@app.get("/api/health")
def health(request: Request) -> dict:
    return {"status": "ok", "llm_configured": request.app.state.agent is not None,
            "documents": len(request.app.state.store.list_documents())}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
