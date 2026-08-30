from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .config import RunConfig
from .terminal import print_startup_info, print_gen_header, print_gen_metadata, print_gen_results, print_final_summary
from . import integrations
from worst_shinka.llm import select_models_for_mode, validate_openrouter_setup
from worst_shinka.brainstorming_system.brainstorm import BrainstormingPipeline, BrainstormResult, EvolutionWorkflow
from worst_shinka.llm.selector import Selector_LLM
import os
import sys
import yaml
from pathlib import Path
from worst_shinka.parent_selector.parents_selector import Selector_Parents
log = logging.getLogger(__name__)
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


def _generation_numbers(run_dir: Path) -> list[int]:
    numbers = []
    for path in run_dir.glob("gen_*"):
        metrics_file = path / "metrics.json"
        if (
            not path.is_dir()
            or not metrics_file.is_file()
            or metrics_file.stat().st_size == 0
        ):
            continue
        try:
            numbers.append(int(path.name.removeprefix("gen_")))
        except ValueError:
            continue
    return sorted(numbers)


def _initial_lineage(config: RunConfig) -> list[dict[str, Any]]:
    return [{
        "id": "model-0",
        "parent_id": None,
        "generation": 0,
        "model": config.initial_model,
        "average_training_score": None,
        "score": 0.0,
        "status": "initial-placeholder"
    }]


def _generation_result_row(candidate: dict[str, Any], *, generation: int, generation_cost: Any = None) -> dict[str, Any]:
    return {
        "generation": candidate.get("generation", generation),
        "status": candidate.get("status") or "pending",
        "average_training_score": candidate.get("average_training_score"),
        "score": candidate.get("score"),
        "cost": generation_cost if generation_cost is not None else candidate.get("cost"),
        "elo": candidate.get("elo"),
        "time": candidate.get("time", candidate.get("duration_seconds")),
    }


def run_evolution(config: RunConfig) -> Path:
    config.validate()
    results_root = config.results_dir.expanduser().resolve()
    run_dir = results_root / (config.name.strip() if config.name is not None else _default_run_name())
    is_continuation = run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    integrations.configure_run(run_dir)
    print_startup_info(config, run_dir)

    if is_continuation:
        log.warning("⚠️ Catalog already existed at %s; continuing with the next generation", run_dir)

    models_path = run_dir / "models.json"
    previous_model_ids: tuple[str, ...] = ()
    if models_path.exists():
        previous_selection = json.loads(models_path.read_text(encoding="utf-8"))
        previous_mode = previous_selection.get("mode")
        previous_models = previous_selection.get("models", [])
        previous_model_ids = tuple(
            item["id"] for item in previous_models if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
        if previous_mode != config.mode:
            log.info("Model mode changed from %s to %s; selecting a new model pool", previous_mode, config.mode)
            previous_model_ids = ()

    validation_path = validate_openrouter_setup(config.resolved_models(), output_path=run_dir / "model-validation.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    selected_models = select_models_for_mode(validation["results"], config.mode, preferred_model_ids=previous_model_ids)
    _write_json(
        models_path,
        {"mode": config.mode, "count": len(selected_models), "models": selected_models},
    )

    validated_models = tuple(item["id"] for item in selected_models)
    kept_models = [model_id for model_id in previous_model_ids if model_id in validated_models]
    if previous_model_ids:
        log.info("ℹ️ Kept %s previously selected models: %s", len(kept_models), ", ".join(kept_models) or "none")
    log.info("ℹ️ Selected %s valid models: %s", len(validated_models), ", ".join(validated_models))

    lineage_path = run_dir / "lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8")).get("nodes", []) if lineage_path.exists() else _initial_lineage(config)
    generations = _generation_numbers(run_dir)
    created_initial_gen = False
    total_cost = 0.0
    if not generations:
        gen_dir = run_dir / "gen_0"
        gen_dir.mkdir(exist_ok=True)
        # _write_json(gen_dir / "metrics.json", {"generation": 0, "status": "initial-placeholder"})
        initial_src = Path(config.initial_model).expanduser()
        if initial_src.is_dir():
            integrations.prepare_initial_generation(source_dir=initial_src, generation_dir=gen_dir)
            initial_candidates = integrations.train_and_evaluate(
                proposals=[{"generation": 0, "generation_dir": str(gen_dir)}], workers=1,
                tournament=config.tournament
            )
            if initial_candidates:
                lineage = initial_candidates
        initial_node = lineage[0] if lineage else {}
        _write_json(gen_dir / "metrics.json", {
            "generation": 0,
            "status": initial_node.get("status", "initial-placeholder"),
            "average_training_score": initial_node.get("average_training_score"),
            "score": initial_node.get("score"),
            "elo": initial_node.get("elo")
        })
        _write_json(gen_dir / "solutions.json", {"generation": 0, "solutions": lineage})
        _write_json(lineage_path, {"nodes": lineage})
        generations = [0]
        created_initial_gen = True
    if created_initial_gen:
        initial = lineage[0] if lineage else {}
        initial_status = initial.get("status")
        if initial_status not in {"correct", "incorrect"}:
            initial_status = "pending"

        print_gen_results(
            [{
                "generation": 0,
                "status": initial_status,
                "average_training_score": initial.get("average_training_score", "-") if initial.get("average_training_score") is not None else "-",
                "score": initial.get("score", "-") if initial.get("score") is not None else "-",
                "cost": initial.get("cost", "-"),
                "elo": initial.get("elo", "-"),
                "time": initial.get("time", "-")
            }],
            generation=0,
            heading="INITIAL GENERATION"
        )
   


    manifest_path = run_dir / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "config": config.to_dict(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    manifest.update({"status": "running", "last_generation": max(generations)})
    _write_json(manifest_path, manifest)

    last_existing_generation = max(generations)
    first_generation = last_existing_generation + 1
    total_generations = last_existing_generation + config.generations
    if is_continuation:
        for generation in range(1, first_generation):
            log.info(
                "Generation %s/%s - already exists, skipping",
                generation,
                total_generations,
            )
    llm_selector = Selector_LLM(validated_models)
    parent_selector = Selector_Parents()
    for generation in range(first_generation, first_generation + config.generations):

        gen_dir = run_dir / f"gen_{generation}"
        gen_dir.mkdir(exist_ok=True)

        evolution_models = llm_selector.select_models()
        log.info("Evolving proposals using models: %s", ", ".join(evolution_models))

        config_path = run_dir / f"gen_{generation}"
        rl_modules = integrations._rl_modules()
        train = rl_modules["train"].train
        workflow = EvolutionWorkflow(models=evolution_models,
                                     gen_id=generation,
                                     history_path=str(config_path / "brainstorming"),
                                     train_function=train,
                                     max_debate_rounds=1,
                                     workers=config.workers)
        parent_data = integrations.get_parents_data(run_dir)
        parent_by_gen = {data["gen"]: data for data in parent_data}
        performances = []
        for data in parent_data:
            value = data["metrics"].get("score")
            if value is None:
                value = data["metrics"].get("elo", 0.0)
            performances.append(value)
        ids = [data["gen"] for data in parent_data]
        parent_count = min(generation, config.parents, len(parent_data))
        selected_parents = parent_selector.select_parent_ids(parent_count, ids, performances)
        print_gen_header(generation=generation)
        print_gen_metadata(generation=generation, name = run_dir.name,
                           parent_ids=[str(parent_id) for parent_id in selected_parents], mode = config.mode)
        log.info("Generation %s/%s", generation, total_generations)
        log.info("Selected parents for generation %s: %s", generation, ", ".join(selected_parents))
        os.environ["WORST_SHINKA_PARENT_GENERATIONS"] = ",".join(selected_parents)
        result = workflow.execute_crossover([parent_by_gen[gen_id] for gen_id in selected_parents])
        if not result:
            log.warning("No accepted evolution proposal for generation %s - continuing", generation)
            generation_cost = workflow.cost_usd
            if generation_cost is not None:
                total_cost += generation_cost
            selected_parent_model_ids = [
                f"model-{str(parent_id).removeprefix('gen_')}"
                for parent_id in selected_parents
            ]
            failed_node = {
                "id": f"model-{generation}",
                "generation": generation,
                "parent_id": selected_parent_model_ids[0] if selected_parent_model_ids else None,
                "parent_ids": selected_parent_model_ids,
                "status": "incorrect",
                "score": None,
                "elo": None,
                "models": evolution_models,
                "cost": generation_cost,
                "failure_reason": "No proposal was accepted by the evolution judge",
            }
            lineage.append(failed_node)
            _write_json(gen_dir / "metrics.json", {
                "generation": generation,
                "status": "incorrect",
                "score": None,
                "elo": None,
                "cost": generation_cost,
                "failure_reason": failed_node["failure_reason"],
            })
            _write_json(gen_dir / "solutions.json", {
                "generation": generation,
                "models": evolution_models,
                "parents": selected_parent_model_ids,
                "evaluated": [],
                "accepted": [],
                "status": "incorrect",
                "failure_reason": failed_node["failure_reason"],
            })
            _write_json(gen_dir / "lineage.json", {"nodes": lineage})
            _write_json(lineage_path, {"nodes": lineage})
            print_gen_results(
                [{"generation": generation, "status": "incorrect", "cost": generation_cost if generation_cost is not None else "-"}],
                generation=generation,
            )
            continue
        parent_selector.update_N(selected_parents)
        log.info("Generation %s: brainstorming winner is %s", generation, result["winner_id"])
        (gen_dir / "algorithm.py").write_text(result["winner_code"], encoding="utf-8")
        (gen_dir / "config.yaml").write_text(
            yaml.safe_dump(result["winner_config"], sort_keys=False), encoding="utf-8"
        )
        proposals = [{"generation": generation, "generation_dir": str(gen_dir)}]
        log.info("Training & evaluating %s proposal(s) (workers=%s)...", len(proposals), config.workers)
        evaluated = integrations.train_and_evaluate(
            proposals=proposals,
            workers=config.workers,
            tournament=config.tournament,
        )
        log.info("Recording %s trained candidate(s)...", len(evaluated))
        accepted = evaluated
        selected_parent_model_ids = [
            f"model-{str(parent_id).removeprefix('gen_')}"
            for parent_id in selected_parents
        ]
        for candidate in evaluated:
            candidate["parent_ids"] = selected_parent_model_ids
            candidate["parent_id"] = selected_parent_model_ids[0] if selected_parent_model_ids else None
        # A full tournament refreshes ratings for every old generation too.
        for node in lineage:
            node_generation = node.get("generation")
            if node_generation is None:
                continue
            metrics_file = run_dir / f"gen_{node_generation}" / "metrics.json"
            if not metrics_file.is_file():
                continue
            refreshed = json.loads(metrics_file.read_text(encoding="utf-8"))
            for key in ("wins", "losses", "draws", "matches", "win_rate", "score", "elo"):
                if key in refreshed:
                    node[key] = refreshed[key]
        lineage.extend(accepted)
        generation_cost = workflow.cost_usd
        total_cost += generation_cost or 0.0

        accepted_ids = [item.get("id") for item in accepted]
        result_rows = []

        for candidate in evaluated:
            candidate_cost = candidate.get("cost", candidate.get("cost_usd"))
            try:
                total_cost += float(candidate_cost or 0)
            except(TypeError, ValueError):
                pass

            candidate_id = candidate.get("id")
            status = candidate.get("status")
            if status not in {"correct", "incorrect"}:
                accepted_candidate = candidate in accepted or (
                    candidate_id is not None and candidate_id in accepted_ids)
                status = "correct" if accepted_candidate else "incorrect"
            result_rows.append(_generation_result_row({
                **candidate,
                "status": status,
                "cost": generation_cost if generation_cost is not None else candidate.get("cost", "-"),
                "generation": generation,
            }, generation=generation, generation_cost=generation_cost))
        log.info("Generation %s complete — %s/%s candidate(s) accepted", generation, len(accepted), len(evaluated))
        print_gen_results(result_rows, generation=generation)
        winner_candidate = evaluated[0] if evaluated else {}
        _write_json(gen_dir / "metrics.json", {
            "generation": generation,
            "status": winner_candidate.get("status", "pending"),
            "score": winner_candidate.get("score"),
            "elo": winner_candidate.get("elo"),
            "wins": winner_candidate.get("wins"),
            "losses": winner_candidate.get("losses"),
            "draws": winner_candidate.get("draws"),
            "matches": winner_candidate.get("matches"),
            "win_rate": winner_candidate.get("win_rate"),
        })
        _write_json(gen_dir / "solutions.json", {
            "generation": generation,
            "models": evolution_models,
            "proposals": proposals,
            "evaluated": evaluated,
            "accepted": accepted,
        })
        _write_json(gen_dir / "lineage.json", {"nodes": lineage})
        _write_json(lineage_path, {"nodes": lineage})

        who_won = 1 if result["winner_id"] == "proposal_1" else 2
        llm_selector.update_probabilities(evolution_models[0], evolution_models[1], who_won)
        log.info(
            "Updated model selection probabilities (winner: %s)",
            evolution_models[0] if who_won == 1 else evolution_models[1],
        )

    manifest.update({
        "status": "integration-placeholders-completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "last_generation": max(_generation_numbers(run_dir)),
        "note": "The CLI loop ran, placeholders to replace",
    })

    _write_json(manifest_path, manifest)

    log.info("Evolution finished — %s model(s) produced across %s generation(s), total cost $%.4f",
              len(lineage), max(_generation_numbers(run_dir)) + 1, total_cost)
    print_final_summary(lineage)

    return run_dir