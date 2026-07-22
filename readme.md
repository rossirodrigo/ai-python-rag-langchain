# RAG with Gemini + LangChain

Retrieval-augmented generation pipeline using LangChain, Google Gemini, and Chroma.

## Structure

```
src/rag/
  config.py      # env vars, model names, chunking/retrieval settings
  loaders.py      # file loading (pdf/txt/md/docx) + text splitting
  embeddings.py    # Gemini embeddings client
  llm.py           # Gemini chat model client
  vectorstore.py   # Chroma persistence + document upsert
  chain.py         # history-aware RAG chain
scripts/
  ingest.py        # upload/add files to the vector store
  chat.py          # interactive conversation over ingested docs
data/
  raw/             # source documents (gitignored)
  chroma_db/       # persisted vector store (gitignored)
```

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY
```

## Usage

Ingest documents:

```bash
python scripts/ingest.py data/raw/mydoc.pdf
python scripts/ingest.py --dir data/raw
```

Chat with the ingested documents:

```bash
python scripts/chat.py
```
