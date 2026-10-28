import zipfile

from pypdf import PdfWriter
import pytest

from backend.ingestion import load_and_split
from backend.service import RAGService
from test_api import Generator, config
from test_index_recovery import MemoryStore


def test_extracted_text_limit_and_csv_ingestion(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("name,value\nRedwood,7319\n")
    _, chunks = load_and_split(path, "table.csv", config(tmp_path))
    assert "Redwood" in chunks[0].page_content
    with pytest.raises(ValueError, match="character limit"):
        load_and_split(path, "table.csv", config(tmp_path, max_document_chars=5))


def test_pdf_page_and_docx_expansion_limits(tmp_path):
    pdf = tmp_path / "two.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    with pdf.open("wb") as output:
        writer.write(output)
    with pytest.raises(ValueError, match="page limit"):
        load_and_split(pdf, "two.pdf", config(tmp_path, max_document_pages=1))
    docx = tmp_path / "expanded.docx"
    with zipfile.ZipFile(docx, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 10000)
    with pytest.raises(ValueError, match="expanded size"):
        load_and_split(docx, "expanded.docx", config(tmp_path, max_upload_bytes=1024))


def test_second_service_cannot_mutate_a_live_catalog(tmp_path):
    settings = config(tmp_path)
    first = RAGService(settings, MemoryStore(), Generator())
    try:
        with pytest.raises(RuntimeError, match="one worker"):
            RAGService(settings, MemoryStore(), Generator())
    finally:
        first.close()
    next_service = RAGService(settings, MemoryStore(), Generator())
    next_service.close()
