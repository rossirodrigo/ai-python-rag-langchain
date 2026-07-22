from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from rag.config import RETRIEVER_K
from rag.llm import get_llm
from rag.vectorstore import get_vectorstore

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can "
    "be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

QA_PROMPT = (
    "You are an assistant answering questions using only the retrieved "
    "context below. If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}"
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain() -> RunnableLambda:
    """Builds a history-aware RAG runnable.

    LangChain 1.x removed the legacy `langchain.chains` helpers
    (create_retrieval_chain, create_history_aware_retriever), so this is
    assembled directly from LangChain Expression Language (LCEL) primitives.
    """
    llm = get_llm()
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    condense_question_chain = contextualize_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    answer_chain = qa_prompt | llm | StrOutputParser()

    def invoke(inputs: dict) -> dict:
        question = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        if chat_history:
            question = condense_question_chain.invoke(
                {"input": question, "chat_history": chat_history}
            )

        docs = retriever.invoke(question)
        answer = answer_chain.invoke(
            {
                "input": inputs["input"],
                "chat_history": chat_history,
                "context": _format_docs(docs),
            }
        )
        return {"answer": answer, "context": docs}

    return RunnableLambda(invoke)
