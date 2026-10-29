import io

import pytest
from pypdf import PdfWriter

from app.ingest import IngestError, Page, chunk_pages, extract_pages


def test_extract_text_file():
    pages = extract_pages("notes.md", "# Title\nSome text".encode())
    assert len(pages) == 1 and pages[0].number is None


@pytest.mark.parametrize("filename, data, message", [
    ("virus.exe", b"MZ", "Unsupported"),
    ("empty.txt", b"   \n", "No text"),
    ("latin1.txt", "café".encode("latin-1"), "UTF-8"),
    ("broken.pdf", b"not a pdf", "Could not read PDF"),
])
def test_extract_rejects_bad_files(filename, data, message):
    with pytest.raises(IngestError, match=message):
        extract_pages(filename, data)


def test_pdf_without_text_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(IngestError, match="No text"):
        extract_pages("scan.pdf", buffer.getvalue())


def test_chunks_overlap_and_keep_page_numbers():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_pages([Page(3, " ".join(words))], chunk_words=10, overlap_words=2)
    assert [c.text.split()[0] for c in chunks] == ["w0", "w8", "w16"]
    assert chunks[-1].text.split()[-1] == "w24"
    assert all(c.page == 3 for c in chunks)


def test_short_page_is_one_chunk():
    assert len(chunk_pages([Page(None, "just a few words")], chunk_words=10, overlap_words=2)) == 1


def test_search_finds_relevant_chunk(filled_store):
    hits = filled_store.search("meals travel per day", top_k=3)
    assert "Meals during business travel" in hits[0].text
    assert hits[0].document == "handbook.md"


def test_keyword_search_catches_exact_terms(filled_store):
    hits = filled_store.search("AES-256", top_k=2)
    assert hits[0].document == "product.pdf" and hits[0].page == 2


def test_unrelated_query_returns_nothing(filled_store):
    assert filled_store.search("zebra quantum volcano") == []


def test_fts_syntax_in_query_is_safe(filled_store):
    filled_store.search('meals" OR NEAR( * -')  # must not raise an FTS5 syntax error


def test_delete_document_removes_its_chunks(filled_store):
    product_id = next(d["id"] for d in filled_store.list_documents() if d["name"] == "product.pdf")
    assert filled_store.delete_document(product_id)
    assert all(h.document != "product.pdf" for h in filled_store.search("backups encrypted AES"))
    assert not filled_store.delete_document(product_id)


def test_empty_store_search(store):
    assert store.search("anything") == []
