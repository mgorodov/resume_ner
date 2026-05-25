from typing import Any

from pathlib import Path

import spacy
from app.config.settings import settings
from app.ml.schema import NamedEntity

from loguru import logger


class NERService:
    def __init__(self, model_dir: Path = settings.ml.model_dir):
        self._model_dir = model_dir
        logger.debug(f"Loading model from {model_dir}")
        self._nlp = spacy.load(self._model_dir)

    def infer(self, text: str) -> list[NamedEntity]:
        doc = self._nlp(text)
        return [
            NamedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        ]

    def get_model_debug_info(self) -> dict[str, Any]:
        return {
            "pipeline": self._nlp.pipe_names,
            "labels": list(self._nlp.get_pipe("ner").labels),
        }
