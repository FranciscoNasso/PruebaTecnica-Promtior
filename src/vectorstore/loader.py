from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import settings
import re

def get_vectorstore() -> Chroma:
    """
    Carga el vector store persistido en el directorio configurado.
    Si la clave OpenAI no funciona por cuota/errores, usa sentence-transformers local.
    Si hay mismatch de dimensiones, detecta y ajusta la función de embeddings.
    """
    vector_dir = settings.vectorstore_dir

    # helpers para embeddings
    def make_local():
        from sentence_transformers import SentenceTransformer
        class LocalEmbeddings:
            def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)
            def embed_documents(self, texts):
                return self.model.encode(texts, show_progress_bar=False).tolist()
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        return LocalEmbeddings()

    def make_openai():
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)

    # Determine initial embeddings preference
    embeddings = None
    use_local = False
    if settings.openai_api_key:
        try:
            embeddings = make_openai()
            # quick network test
            try:
                embeddings.embed_documents(["test"])
                settings.openai_usable = True
            except Exception:
                embeddings = None
                use_local = True
                settings.openai_usable = False
        except Exception:
            embeddings = None
            use_local = True
            settings.openai_usable = False
    else:
        use_local = True
        settings.openai_usable = False

    if use_local or embeddings is None:
        try:
            embeddings = make_local()
            settings.openai_usable = False
        except Exception as e:
            raise RuntimeError("OpenAI no disponible y sentence-transformers no instalado. Instalar: pip install sentence-transformers") from e

    # Crear Chroma con la función elegida
    vectorstore = Chroma(persist_directory=str(vector_dir), embedding_function=embeddings)

    # Prueba de compatibilidad: similarity_search corta para detectar mismatch de dimensión
    try:
        vectorstore.similarity_search("test", k=1)
        return vectorstore
    except Exception as e:
        msg = str(e)
        # intentar detectar patrón "expecting embedding with dimension of X, got Y"
        m = re.search(r"expecting embedding with dimension of\s+(\d+).*got\s+(\d+)", msg, flags=re.IGNORECASE)
        if m:
            expected_dim = int(m.group(1))
            got_dim = int(m.group(2))
            print(f"Detected embedding dimension mismatch: expected={expected_dim}, got={got_dim}. Adjusting embeddings...")
            # heurística de modelos: 384 -> all-MiniLM-L6-v2 (local), 1536 -> OpenAI text-embedding-3-small
            if expected_dim == 384:
                # switch to local
                try:
                    local = make_local()
                except Exception:
                    raise RuntimeError("Colección requiere embeddings dim=384 pero sentence-transformers no está instalado.")
                # reconstruir Chroma con local embeddings
                vectorstore = Chroma(persist_directory=str(vector_dir), embedding_function=local)
                settings.openai_usable = False
                return vectorstore
            elif expected_dim == 1536:
                # switch to OpenAI
                if not settings.openai_api_key:
                    raise RuntimeError("Colección requiere embeddings dim=1536 (OpenAI) pero OPENAI_API_KEY no está configurada.")
                openai_emb = make_openai()
                vectorstore = Chroma(persist_directory=str(vector_dir), embedding_function=openai_emb)
                settings.openai_usable = True
                return vectorstore
            else:
                # no sabemos qué modelo usar para esa dimensión
                raise RuntimeError(f"Embedding dimension mismatch and unknown expected dim={expected_dim}. Re-create vectorstore with desired embeddings or migrate data.")
        # si no se pudo parsear, relanzar
        raise
