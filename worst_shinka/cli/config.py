from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_INITIAL_MODEL = "tbd-Dawid"


@dataclass(frozen=True)
class RunConfig:
    models: tuple[str, ...]
    results_dir: Path
    initial_model: str | Path = DEFAULT_INITIAL_MODEL
    generations: int = 10
    workers: int = 1
    parents: int = 1

    def validate(self) -> "RunConfig":
        cleaned_models = [model.strip() for model in self.models if model.strip()]
        if len(cleaned_models) < 2:
            raise ValueError("models must contain at least two model identifiers")
        if len(set(cleaned_models)) != len(cleaned_models):
            raise ValueError("models must not contain duplicates")
        if not str(self.initial_model).strip():
            raise ValueError("initial_model cannot be empty")
        for name in ("generations", "workers", "parents"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["results_dir"] = str(self.results_dir)
        return result

    