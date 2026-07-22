from langchain_openai import OpenAIEmbeddings

from rag.config import EMBEDDING_MODEL, LMSTUDIO_BASE_URL


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=LMSTUDIO_BASE_URL,
        api_key="lm-studio",
        check_embedding_ctx_length=False,
    )
