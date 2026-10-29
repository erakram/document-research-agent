"""Turn uploaded files into text chunks ready for embedding.

Each chunk keeps its page number, so answers can cite "handbook.pdf, page 3".
Chunks overlap a little so a sentence split across a boundary is still found.
"""

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_TYPES = {".pdf", ".txt", ".md"}


class IngestError(ValueError):
    """The file can't be used (wrong type, unreadable, or no text)."""


@dataclass
class Page:
    number: int | None  # None for plain-text files, which have no pages
    text: str


@dataclass
class Chunk:
    page: int | None
    text: str


def extract_pages(filename: str, data: bytes) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise IngestError(f"Unsupported file type '{suffix}'. Use PDF, TXT or MD.")

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [Page(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
        except Exception as e:
            raise IngestError(f"Could not read PDF: {e}") from e
    else:
        try:
            pages = [Page(None, data.decode("utf-8"))]
        except UnicodeDecodeError as e:
            raise IngestError("Text files must be UTF-8 encoded.") from e

    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise IngestError("No text found. Scanned PDFs (images) need OCR first.")
    return pages


def chunk_pages(pages: list[Page], chunk_words: int = 150, overlap_words: int = 30) -> list[Chunk]:
    """Split each page into windows of ~chunk_words words, overlapping by overlap_words."""
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")
    chunks = []
    step = chunk_words - overlap_words
    for page in pages:
        words = page.text.split()
        for start in range(0, len(words), step):
            window = words[start:start + chunk_words]
            chunks.append(Chunk(page.number, " ".join(window)))
            if start + chunk_words >= len(words):
                break
    return chunks
