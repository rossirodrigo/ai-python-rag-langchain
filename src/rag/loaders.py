from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE

LOADERS_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    loader_cls = LOADERS_BY_SUFFIX.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader_cls(str(file_path)).load()


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def load_and_split(file_path: Path) -> list[Document]:
    return split_documents(load_file(file_path))
