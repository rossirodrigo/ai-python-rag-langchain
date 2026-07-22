#!/usr/bin/env python
"""Upload files into the Chroma vector store.

Usage:
    python scripts/ingest.py path/to/file1.pdf path/to/file2.txt
    python scripts/ingest.py --dir data/raw
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.loaders import LOADERS_BY_SUFFIX, load_and_split
from rag.vectorstore import add_documents


def collect_files(paths: list[str], directory: str | None) -> list[Path]:
    files = [Path(p) for p in paths]
    if directory:
        dir_path = Path(directory)
        files += [
            f for f in dir_path.rglob("*") if f.suffix.lower() in LOADERS_BY_SUFFIX
        ]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Files to ingest")
    parser.add_argument("--dir", help="Directory to scan for supported files")
    args = parser.parse_args()

    files = collect_files(args.files, args.dir)
    if not files:
        parser.error("No files provided. Pass file paths or --dir <directory>.")

    for file_path in files:
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue
        chunks = load_and_split(file_path)
        add_documents(chunks)
        print(f"Ingested {file_path} -> {len(chunks)} chunks")


if __name__ == "__main__":
    main()
