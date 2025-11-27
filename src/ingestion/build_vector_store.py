from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from .load_promtior_site import get_promtior_documents


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_vectorstore_dir() -> Path:
    return get_project_root() / "data" / "vectorstore"


def build_vector_store() -> None:
    """
    - Carga documentos de Promtior (web + PDF opcional)
    - Los divide en chunks
    - Crea y persiste un Chroma vector store
    """
    print("Cargando documentos de Promtior...")
    docs = get_promtior_documents()
    print(f"Total de documentos originales: {len(docs)}")

    if not docs:
        print("No se cargaron documentos. Revisá las URLs o el PDF.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Total de chunks después de split: {len(split_docs)}")

    # Embeddings (usa OPENAI_API_KEY del entorno)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # podés cambiar el modelo si querés
    )

    vectorstore_dir = get_vectorstore_dir()
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    print(f"Construyendo y guardando vector store en: {vectorstore_dir}")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(vectorstore_dir),
    )

    # Chroma ya persiste en from_documents cuando se le pasa persist_directory
    # pero por las dudas:
    vectorstore.persist()

    print("Vector store creado y persistido correctamente.")


if __name__ == "__main__":
    build_vector_store()
