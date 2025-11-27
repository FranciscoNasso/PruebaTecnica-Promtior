from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.schema import Document


# URLs de Promtior que querés indexar
PROMTIOR_URLS = [
    "https://www.promtior.ai/",
    "https://www.promtior.ai/use-cases",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/contacto",
    "https://www.promtior.ai/blog",
]


def get_project_root() -> Path:
    """
    Devuelve la raíz del proyecto asumiendo:
    src/ingestion/este_archivo.py
    """
    return Path(__file__).resolve().parents[2]


def get_raw_data_dir() -> Path:
    return get_project_root() / "data" / "raw"


def load_promtior_web_pages(urls: Optional[List[str]] = None) -> List[Document]:
    """
    Carga el contenido de las páginas web de Promtior usando WebBaseLoader.
    """
    if urls is None:
        urls = PROMTIOR_URLS

    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs


def load_promtior_presentation() -> List[Document]:
    """
    Carga la presentación de Promtior en PDF (si existe).
    Ruta esperada: data/raw/promtior_presentation.pdf
    """
    raw_dir = get_raw_data_dir()
    pdf_path = raw_dir / "AI Engineer-Tecnical-Test.pdf"

    if not pdf_path.exists():
        # Si no existe, devolvemos lista vacía para no romper el flujo
        return []

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    return docs


def get_promtior_documents(
    extra_urls: Optional[List[str]] = None,
    include_presentation: bool = True,
) -> List[Document]:
    """
    Función principal que usará el resto del código:
    - Carga las páginas web de Promtior
    - Opcionalmente, carga la presentación en PDF
    - Devuelve todos los documentos juntos
    """
    urls = PROMTIOR_URLS.copy()
    if extra_urls:
        urls.extend(extra_urls)

    web_docs = load_promtior_web_pages(urls)
    pdf_docs: List[Document] = []

    if include_presentation:
        pdf_docs = load_promtior_presentation()

    all_docs = web_docs + pdf_docs

    # Podés normalizar metadatos si querés
    for d in all_docs:
        d.metadata.setdefault("source_type", "web_or_pdf")

    return all_docs


if __name__ == "__main__":
    docs = get_promtior_documents()
    print(f"Documentos cargados: {len(docs)}")
    if docs:
        print("Ejemplo de contenido:\n")
        print(docs[0].page_content[:1000])
