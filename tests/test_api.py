import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent import AgentReply, Citation
from app.main import app
from app.store import DocumentStore
from tests.conftest import FakeEmbedder


class StubAgent:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def ask(self, question, history):
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("SEED_SAMPLE_DOCS", "true")
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.01")  # ~10 KB, to test the size limit
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app.state.store = DocumentStore(tmp_path / "api.db", FakeEmbedder())  # skip the real model
    with TestClient(app) as c:
        yield c
    del app.state.store


def test_sample_documents_are_seeded(client):
    names = {d["name"] for d in client.get("/api/documents").json()}
    assert names == {"northwind-employee-handbook.md", "cloudvault-product-guide.md"}
    assert client.get("/api/health").json()["documents"] == 2


def test_upload_list_delete(client):
    res = client.post("/api/documents", files={"file": ("../../etc/notes.txt", b"Parking is free on Fridays.")})
    assert res.status_code == 201
    doc = res.json()
    assert doc["name"] == "notes.txt" and doc["chunks"] == 1  # path stripped from the name
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404


@pytest.mark.parametrize("name, data, status", [
    ("malware.exe", b"MZ", 400),
    ("big.txt", b"x " * 10_000, 413),
])
def test_upload_rejects_bad_files(client, name, data, status):
    assert client.post("/api/documents", files={"file": (name, data)}).status_code == status


def test_ask_without_api_key_returns_503(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 503


def test_ask_returns_answer_and_citations(client):
    client.app.state.agent = StubAgent(AgentReply(
        "Meals are covered up to USD 60 [S3].",
        [Citation("S3", "handbook.md", None, "Meals ... USD 60 per day.")],
        [{"name": "search_documents", "arguments": {"query": "meals"}}], 2))
    body = client.post("/api/ask", json={"question": "Meal allowance?"}).json()
    assert body["citations"][0] == {"source_id": "S3", "document": "handbook.md", "page": None,
                                    "text": "Meals ... USD 60 per day."}
    assert body["tool_calls"][0]["arguments"]["query"] == "meals"


def test_ask_with_no_documents(client):
    client.app.state.agent = StubAgent(AgentReply("unused"))
    for doc in client.get("/api/documents").json():
        client.delete(f"/api/documents/{doc['id']}")
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 400


def test_ask_validates_input(client):
    client.app.state.agent = StubAgent(AgentReply("unused"))
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "q", "history": [{"role": "system", "content": "x"}]}).status_code == 422


def test_rate_limit_maps_to_429(client):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    client.app.state.agent = StubAgent(error=groq.RateLimitError("slow down", response=response, body=None))
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 429
