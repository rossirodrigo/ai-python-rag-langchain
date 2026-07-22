from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import CHAT_MODEL, GOOGLE_API_KEY


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
