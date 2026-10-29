import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from app.embeddings import normalize
from app.ingest import Chunk
from app.store import DocumentStore


class FakeEmbedder:
    """Deterministic bag-of-words embedding: texts sharing words get similar vectors."""

    dims = 128

    def embed(self, texts):
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1
        return normalize(vectors + 1e-6)


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "test.db", FakeEmbedder())


@pytest.fixture
def filled_store(store):
    store.add_document("handbook.md", [
        Chunk(None, "Meals during business travel are covered up to USD 60 per day."),
        Chunk(None, "Employees may work remotely up to three days per week with manager approval."),
        Chunk(None, "Every employee has a learning budget of USD 1,500 per year."),
    ])
    store.add_document("product.pdf", [
        Chunk(1, "The Business plan costs USD 15 per device per month and keeps backups for one year."),
        Chunk(2, "Backups are encrypted with AES-256 before leaving the device."),
    ], pages=2)
    return store


def tool_call(call_id, name, arguments):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


def llm_message(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeGroq:
    """Stands in for the Groq client: returns scripted responses and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)
