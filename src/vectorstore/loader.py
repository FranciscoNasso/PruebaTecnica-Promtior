from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import settings
import re

def get_vectorstore() -> Chroma:
    """
    Carga el vector store persistido en el directorio configurado.
    Usa exclusivamente OpenAI embeddings. Si no está la clave OPENAI_API_KEY
    o hay un mismatch de dimensiones que no puede resolverse con OpenAI,
    lanza RuntimeError con instrucciones.
    """
    vector_dir = settings.vectorstore_dir

    def make_openai():
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY no configurada. Configurá la variable de entorno "
                "OPENAI_API_KEY para usar embeddings de OpenAI en producción."
            )
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)

    # Inicializar OpenAI embeddings (fallará si no hay clave)
    try:
        embeddings = make_openai()
        # prueba rápida de conectividad/funcionalidad
        try:
            embeddings.embed_documents(["test"])
            settings.openai_usable = True
        except Exception:
            settings.openai_usable = False
            raise RuntimeError("No se pudo generar embeddings con OpenAI. Revisá OPENAI_API_KEY y la conectividad.")
    except Exception as e:
        # Propagar error con mensaje claro
        raise

    # Crear Chroma con la función elegida
    vectorstore = Chroma(persist_directory=str(vector_dir), embedding_function=embeddings)

    # Prueba de compatibilidad: similarity_search corta para detectar mismatch de dimensión
    try:
        vectorstore.similarity_search("test", k=1)
        return vectorstore
    except Exception as e:
        msg = str(e)
        m = re.search(r"expecting embedding with dimension of\s+(\d+).*got\s+(\d+)", msg, flags=re.IGNORECASE)
        if m:
            expected_dim = int(m.group(1))
            got_dim = int(m.group(2))
            # Si la colección requiere 1536, OpenAI es la opción adecuada
            if expected_dim == 1536:
                # ya usamos OpenAI; si llegamos aquí puede ser por fallo temporal
                settings.openai_usable = True
                return vectorstore
            # Si se requiere otra dimensión (ej. 384) y no queremos sentence-transformers,
            # indicamos al usuario que debe reconstruir el vectorstore con el modelo adecuado.
            raise RuntimeError(
                f"Mismatch de dimensión de embeddings: la colección espera dim={expected_dim} pero se generaron dim={got_dim}. "
                "Actualmente este despliegue usa exclusivamente OpenAI embeddings. "
                "Para resolverlo, o bien recreá el vectorstore usando OpenAI embeddings (dim=1536), "
                "o reconstruí la colección localmente con sentence-transformers y luego subí la DB resultante."
            )
        # si no se pudo parsear el error, relanzar para diagnóstico
        raise
