# Promtior RAG Chatbot

Pequeño proyecto RAG que expone un endpoint (via LangServe) y un frontend estático para chatear con la información indexada de Promtior.

## Contenido clave
- Servidor FastAPI + LangServe: `src/main.py`
- Cadena RAG: `src/chains/rag_chain.py`
- Carga de documentos web / PDF: `src/ingestion/load_promtior_site.py`
- Construcción del vector store: `src/ingestion/build_vector_store.py`
- Gestión de vectorstore / embeddings: `src/vectorstore/loader.py`
- Frontend estático: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

## Requisitos
- Python 3.10+ (recomendado)
- (Opcional) `OPENAI_API_KEY` si querés usar OpenAI para embeddings/LLM. Si no, el proyecto puede usar `sentence-transformers` localmente.
- Instalar dependencias: `pip install -r requirements.txt`

## Configuración rápida (local)

Sigue estos pasos para ejecutar el proyecto en tu máquina local (PowerShell / Windows indicado):

1. Crear y activar un entorno virtual (desde la raíz del proyecto):

  - Windows (PowerShell):
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

  - Unix / macOS:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2. Instalar dependencias:

  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

3. Configurar variables de entorno (archivo obligatorio):

  El proyecto lee `environments/local.env`. Crea (o edita) `environments/local.env` con al menos:

  ```text
  # environments/local.env
  OPENAI_API_KEY=sk-...        # opcional: si querés usar OpenAI para embeddings/LLM
  MODEL_NAME=gpt-4o-mini       # opcional: modelo por defecto
  VECTORSTORE_DIR=./data/vectorstore
  FRONTEND_DIR=./frontend
  ```

  - Si no quieres usar OpenAI, deja `OPENAI_API_KEY` vacío o no lo pongas; el proyecto puede usar embeddings locales (`sentence-transformers`) como fallback.

4. (Opcional) Reconstruir el vector store:

  Si no existe `data/vectorstore` o querés re-indexar los documentos/PDFs:

  ```powershell
  Remove-Item .\data\vectorstore -Recurse -Force
  python -m src.ingestion.build_vector_store
  ```

  Esto descargará/leerá documentos (web + PDF) y construirá un Chroma vectorstore persistente.

5. Ejecutar la aplicación (desarrollo):

  ```powershell
  uvicorn src.main:app --reload --port 8000
  ```

  - Frontend: http://127.0.0.1:8000/
  - Playground / rutas LangServe: http://127.0.0.1:8000/promtior-rag/playground/
  - Endpoint programático: POST `http://127.0.0.1:8000/promtior-rag/invoke` (JSON `{ "input": "tu pregunta" }`).

## Probar la API desde PowerShell

Ejemplo rápido usando `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/promtior-rag/invoke" -ContentType "application/json" -Body '{"input":"What does Promtior do?"}'
```

## Docker (opcional)

Archivo `Dockerfile` recomendado incluido. Pasos comunes:

1. Build:

```bash
docker build -t promtior-rag:latest -f Dockerfile .
```

2. Run (montar `data` y pasar env):

PowerShell:
```powershell
docker run --rm -p 8000:8000 --env-file .\environments\local.env -v "${PWD}\data:/app/data" promtior-rag:latest
```

Bash/WSL:
```bash
docker run --rm -p 8000:8000 --env-file ./environments/local.env -v "$(pwd)/data:/app/data" promtior-rag:latest
```