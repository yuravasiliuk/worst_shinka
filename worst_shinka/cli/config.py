from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_INITIAL_MODEL = "tbd-Dawid"
MODEL_MODES = ("low", "medium", "high")
SYSTEM_ASSIGNED_MODELS = Path(__file__).with_name("default_models.json")

def load_system_assigned_models(path: Path = SYSTEM_ASSIGNED_MODELS) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except(OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"default system assigned models from {path} cannot be load: {exc}") from exc

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise ValueError(f"{path} must contain a JSON object with a string list named 'models'")

    cleaned = tuple(model.strip() for model in models if model.strip())
    model_ids = list(cleaned)
    if len(cleaned) < 5:
        raise ValueError("system assigned model must contain at least five models")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("system assigned model must not contain any duplicate models")

    return cleaned


@dataclass(frozen=True)
class RunConfig:
    models: tuple[str, ...] | None
    results_dir: Path
    name: str | None = None
    initial_model: str | Path = DEFAULT_INITIAL_MODEL
    generations: int = 10
    workers: int = 1
    parents: int = 1
    mode: str = "medium"

    def validate(self) -> "RunConfig":
        if self.mode not in MODEL_MODES:
            raise ValueError(f"mode must be one of: {', '.join(MODEL_MODES)}")
        if self.models is not None:
            cleaned_models = [model.strip() for model in self.models if model.strip()]
            if len(cleaned_models) < 2:
                raise ValueError("models must contain at least two model identifiers")
            if len(set(cleaned_models)) != len(cleaned_models):
                raise ValueError("models must not contain duplicates")
        if self.name is not None:
            name = self.name.strip()
            if not name or name in {".", ".."} or Path(name).name != name:
                raise ValueError("name must be a non-empty directory name, not a path")
        if not str(self.initial_model).strip():
            raise ValueError("initial_model cannot be empty")
        for name in ("generations", "workers", "parents"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["models"] = list(self.resolved_models())
        result["results_dir"] = str(self.results_dir)
        result["initial_model"] = str(self.initial_model)
        return result

    def resolved_models(self) -> tuple[str, ...]:
        return self.models if self.models is not None else load_system_assigned_models()

    