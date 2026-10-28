import csv
import hashlib
from pathlib import Path
import re
import tempfile
import zipfile

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from backend.config import Settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}


def safe_filename(filename):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    return (cleaned or "uploaded_document")[:150]


def calculate_file_hash(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def save_upload(file: UploadFile, settings: Settings) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Supported file types: PDF, TXT, DOCX, CSV.")
    settings.ensure_dirs()
    with tempfile.NamedTemporaryFile(dir=settings.upload_dir, suffix=suffix, delete=False) as handle:
        target = Path(handle.name)
        total = 0
        try:
            while data := await file.read(1024 * 1024):
                total += len(data)
                if total > settings.max_upload_bytes:
                    raise HTTPException(413, "Upload exceeds the configured byte limit.")
                handle.write(data)
            if total == 0:
                raise HTTPException(400, "Empty upload.")
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    return target


def read_documents(path, settings):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield Document(page_content=path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield Document(page_content="\n".join(f"{key}: {value}" for key, value in row.items()))
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        if len(reader.pages) > settings.max_document_pages:
            raise ValueError("PDF exceeds the configured page limit.")
        for index, page in enumerate(reader.pages):
            yield Document(page_content=page.extract_text() or "", metadata={"page": index})
    elif suffix == ".docx":
        import docx2txt
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > settings.max_upload_bytes * 4:
                raise ValueError("DOCX expanded size exceeds the configured limit.")
        yield Document(page_content=docx2txt.process(path))
    else:
        raise ValueError("Unsupported file type.")


def load_and_split(file_path, original_name, settings):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    total, documents = 0, []
    for document in read_documents(file_path, settings):
        total += len(document.page_content)
        if total > settings.max_document_chars:
            raise ValueError("Extracted text exceeds the configured character limit.")
        documents.append(document)
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No valid text found in the document.")
    file_hash = calculate_file_hash(file_path)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update({"hash": file_hash, "source": safe_filename(original_name), "chunk_index": index})
    return file_hash, chunks
