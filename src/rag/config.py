import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-flash-lite-latest"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
COLLECTION_NAME = "rag_documents"
RETRIEVER_K = 4

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
    )
