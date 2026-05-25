from pathlib import Path

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent


class MLConfig(BaseModel):
    model_dir: Path = PROJECT_ROOT / "resume_ner_model"
