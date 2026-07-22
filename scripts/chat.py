#!/usr/bin/env python
"""Interactive RAG conversation over the ingested documents.

Usage:
    python scripts/chat.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import AIMessage, HumanMessage

from rag.chain import build_rag_chain


def main() -> None:
    rag_chain = build_rag_chain()
    chat_history: list[HumanMessage | AIMessage] = []

    print("RAG chat ready. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            result = rag_chain.invoke({"input": question, "chat_history": chat_history})
        except Exception as exc:
            print(f"\nAssistant: Sorry, something went wrong answering that ({exc}). Please try again.")
            continue

        answer = result["answer"]
        print(f"\nAssistant: {answer}")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()
