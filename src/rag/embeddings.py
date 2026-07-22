from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag.config import EMBEDDING_MODEL, GOOGLE_API_KEY


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
