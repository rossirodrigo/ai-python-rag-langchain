from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import CHROMA_DIR, COLLECTION_NAME
from rag.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def add_documents(documents: list[Document]) -> list[str]:
    vectorstore = get_vectorstore()
    return vectorstore.add_documents(documents)
