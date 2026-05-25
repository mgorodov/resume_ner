from pathlib import Path

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "resume_ner.sqlite3"


class DatabaseConfig(BaseModel):
    url: str = f"sqlite+aiosqlite:///{DB_PATH}"
    echo: bool = True
