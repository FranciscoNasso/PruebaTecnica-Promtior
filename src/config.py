from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path

# Resolver ruta del proyecto de forma robusta
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> parent = proyecto raíz
ENV_PATH = PROJECT_ROOT / "environments" / "local.env"

# Cargar .env sólo si existe, o usar ENV_FILE como override.
# No lanzar excepción si no existe: en Railway las vars vienen del entorno.
env_file_override = os.getenv("ENV_FILE")
if env_file_override:
    env_path = Path(env_file_override)
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
elif ENV_PATH.exists():
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
