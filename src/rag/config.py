import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

CHAT_MODEL = "gemini-flash-lite-latest"

# Embeddings run locally via LM Studio's OpenAI-compatible server.
# Load "text-embedding-embeddinggemma-300m" (Google's EmbeddingGemma) in LM Studio
# and start its local server before running ingest/chat.
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
COLLECTION_NAME = "rag_documents"
RETRIEVER_K = 4

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
    )
