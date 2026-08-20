from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RunConfig
from . import integrations
from worst_shinka.llm import validate_openrouter_setup

log = logging.getLogger(__name__)
#TODO ogarnac zeby mozna bylo wskazac ten sam docelowy katalog jako kontynuacja...
class _PathEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def _write_json(path: Path, value: Any) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False, cls=_PathEncoder) + "\n"
    path.write_text(content, encoding="utf-8")

def _default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f")

def run_evolution(config: RunConfig) -> Path:
    config.validate()
    results_root = config.results_dir.expanduser().resolve()
    run_dir = results_root / (config.name.strip() if config.name is not None else _default_run_name())
    run_dir.mkdir(parents=True, exist_ok=False)
    validate_openrouter_setup(config.resolved_models(), output_path=run_dir / "model-validation.json")

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version" : 1,
        "status" : "running",
        "started_at" : started_at,
        "config": config.to_dict()
    }
    _write_json(run_dir / "run.json", manifest)

    lineage: list[dict[str, Any]] = [{
        "id":"model-0",
        "parent_id": None,
        "generation": 0,
        "model": config.initial_model,
        "score": None,
        "status": "initial-placeholder"
    }]

    for generation in range(1, config.generations + 1):
        log.info("Generation %s/%s", generation, config.generations)

        #placeholder Dawid provide results here
        available = integrations.fetch_candidates(limit=config.parents)
        parents = available[:config.parents]
        selected_models = integrations.select_models_with_bandit(models=config.resolved_models(), count=2)
        proposals = integrations.evolve_with_models(models = selected_models, parents=parents, generation=generation)
        evaluated = integrations.train_and_evaluate(proposals=proposals, workers=config.workers)
        accepted = integrations.judge_candidates(candidates=evaluated)
        lineage.extend(accepted)

        _write_json(run_dir / "lineage.json", {"nodes":lineage})

    manifest.update({
        "Status": "integration-placeholders-completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "note": "The CLI loop ran, placeholders to replace"
    })

    _write_json(run_dir / "run.json", manifest)
    _write_json(run_dir / "lineage.json", {"nodes":lineage})

    return run_dir