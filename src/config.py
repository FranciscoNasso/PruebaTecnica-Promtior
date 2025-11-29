from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path

# Resolver ruta del proyecto de forma robusta
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> parent = proyecto raíz
ENV_PATH = PROJECT_ROOT / "environments" / "local.env"

# Carga el .env (si existe)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH))
else:
    raise FileNotFoundError(f"El archivo de entorno no se encontró en: {ENV_PATH}")

class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "./data/vectorstore")
    frontend_dir: str = os.getenv("FRONTEND_DIR", "./frontend")
    log_level: str = os.getenv("APP_LOG_LEVEL", "info")
    # runtime flag: indica si OpenAI está operativo (true por defecto)
    openai_usable: bool = True

settings = Settings()
