from __future__ import annotations

from pathlib import Path
import time

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.load_promtior_site import get_promtior_documents
from src.config import settings


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_vectorstore_dir() -> Path:
    return get_project_root() / "data" / "vectorstore"


def build_vector_store() -> None:
    """
    - Carga documentos de Promtior
    - Los divide en chunks
    - Crea y persiste un Chroma vector store
    """
    print("Cargando documentos de Promtior...")
    docs, pdf_loaded = get_promtior_documents()
    print(f"Total de documentos originales: {len(docs)}")
    print(f"PDF cargado: {pdf_loaded}")

    if not docs:
        print("No hay documentos para procesar.")
        return

    # Chunking
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
    except Exception as e:
        print("Error al inicializar el text splitter:", e)
        raise

    split_docs = text_splitter.split_documents(docs)
    print(f"Total de chunks después de split: {len(split_docs)}")

    # OpenAI embeddings required (no sentence-transformers fallback)
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY no configurada. Este script requiere embeddings de OpenAI. "
            "Configura la variable de entorno OPENAI_API_KEY o modifica el loader para usar embeddings locales."
        )

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
        print("Usando OpenAIEmbeddings.")
    except Exception as e:
        print("OpenAIEmbeddings falló al inicializar:", e)
        raise RuntimeError("No se pudo inicializar OpenAIEmbeddings. Revisá OPENAI_API_KEY y conectividad.") from e

    vectorstore_dir = get_vectorstore_dir()
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    print(f"Construyendo y guardando vector store en: {vectorstore_dir}")

    # Inicializar Chroma y añadir por batches
    vectorstore = Chroma(persist_directory=str(vectorstore_dir), embedding_function=embeddings)
    batch_size = 8
    total = len(split_docs)
    added = 0
    for start in range(0, total, batch_size):
        batch = split_docs[start : start + batch_size]
        texts = [d.page_content for d in batch]
        metadatas = [getattr(d, "metadata", {}) for d in batch]

        retries = 0
        while True:
            try:
                vectorstore.add_texts(texts=texts, metadatas=metadatas)
                added += len(texts)
                print(f"Batch añadido: {start} - {start + len(texts)} (total añadido: {added}/{total})")
                break
            except Exception as e:
                msg = str(e).lower()
                # Si es un error de cuota/rate-limit -> fallar con mensaje instructivo
                if "insufficient_quota" in msg or "quota" in msg or "429" in msg or "rate limit" in msg:
                    raise RuntimeError(
                        "Error de cuota/rate-limit de OpenAI al añadir embeddings. "
                        "Revisá la clave OPENAI_API_KEY, billing y límites de cuota."
                    ) from e

                retries += 1
                if retries > 5:
                    print("Máximos reintentos alcanzados al añadir batch:", e)
                    raise
                sleep = 2 ** retries
                print(f"Error al añadir batch, reintentando en {sleep}s... (int {retries}) - error: {e}")
                time.sleep(sleep)

    # Persistir
    try:
        vectorstore.persist()
        print("Vectorstore persistido correctamente.")
        print(f"Chunks añadidos: {added}")
    except Exception as e:
        print("Error al persistir vectorstore:", e)
        raise

    print("Vector store creado y persistido correctamente.")


if __name__ == "__main__":
    build_vector_store()
