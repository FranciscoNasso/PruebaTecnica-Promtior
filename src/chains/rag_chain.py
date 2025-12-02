from __future__ import annotations

from typing import List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import openai


from src.config import settings
from src.vectorstore.loader import get_vectorstore

Document = Any


def _format_docs(docs: List[Document]) -> str:
    """
    Junta el contenido de los documentos recuperados en un solo string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# Cargar vector store y crear retriever
_vectorstore = get_vectorstore()
retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

# Definir prompt para el RAG
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions ONLY using the context provided.
If the answer is not in the context, say that you don't know and suggest rephrasing the question.

Context:
{context}

Question:
{question}

Answer in a clear and concise way.
"""
)

# LLM
llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key,
    temperature=0,
)


# RAG chain: input = pregunta (string)
rag_chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)
