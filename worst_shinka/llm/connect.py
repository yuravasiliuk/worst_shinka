from __future__ import annotations

import json
import logging
import random
import socket
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from decimal import Decimal, InvalidOperation
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from worst_shinka.cli.settings import config_dir, get_openrouter_api_key

log = logging.getLogger(__name__)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
VALIDATION_CACHE_NAME = "model-validation.json"
MAX_COST_PER_MILLION = Decimal("35")
MIN_VALID_MODELS = 5
FINAL_MODEL_COUNT = 5
REQIEST_TIMEOUT_SECONDS = 20
ATTEMPT_DELAYS_SECONDS = (10, 20)
MODEL_PROFILES = {
    "low": {"min_context": 8_192, "max_context": 262_144},
    "medium": {"min_context": 131_072, "max_context": 400_000},
    "high": {"min_context": 200_000, "max_context": None},
}

class OpenRouterError(RuntimeError):
    """Raise when Openrouter cannot be used"""

class _RetryableConnectionError(OpenRouterError):
    pass

def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "worst-shinka/0.0.0"
    }

def _error_message(body: bytes, fallback: str) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
        error = payload.get("error", {})
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    except(UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass
    return fallback

def _get_json(path: str, api_key: str) -> dict[str, Any]:
    request = Request(f"{OPENROUTER_API_BASE}{path}", headers=_headers(api_key))
    try:
        with urlopen(request, timeout=REQIEST_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise _RetryableConnectionError(f"‼️ OpenRouter returned HTTP {response.status}")
            payload = json.load(response)
    except HTTPError as exc:
        message = _error_message(exc.read(), str(exc.reason))
        if exc.code in {401, 403}:
            raise OpenRouterError(f"❌ OpenRouter rejected the API key (HTTP {exc.code}): {message}") from exc
        if exc.code == 404:
            raise OpenRouterError(f"OpenRouter endpoint not found: {path}") from exc
        raise _RetryableConnectionError(f"OpenRouter returned HTTP {exc.code}: {message}") from exc
    except(URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise _RetryableConnectionError(f"cannot connect to OpenRouter: {exc}") from exc

    if not isinstance(payload, dict):
        raise OpenRouterError(f"❌ OpenRouter returned an invalid response for {path}")

    return payload


def _connect_with_retries(api_key: str, *, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    last_error: OpenRouterError | None = None
    max_attempts = len(ATTEMPT_DELAYS_SECONDS) + 1
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            delay = ATTEMPT_DELAYS_SECONDS[attempt - 2]
            log.info("⏲ Retrying OpenRouter connection attempt %s/%s in %s seconds", attempt, max_attempts, delay)
            sleep(delay)
        try:
            result = _get_json("/key", api_key)
            log.info("🖧 OpenRouter connection established")
            return result
        except _RetryableConnectionError as exc:
            last_error = exc
            log.error("❌ OpenRouter attempt %s/%s failed: %s", attempt, max_attempts, exc)

    message = f"❌ OpenRouter connection failed after {max_attempts} attempts: {last_error}"
    log.critical(message)
    raise OpenRouterError(message)


def _money(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except(InvalidOperation, ValueError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = str(value).strip().lower()
        multiplier = 1
        if normalized.endswith("t"):
            multiplier = 1_000
            normalized = normalized[:-1]
        elif normalized.endswith("b"):
            normalized = normalized[:-1]
        elif normalized.endswith("m"):
            multiplier = 0.001
            normalized = normalized[:-1]
        number = float(normalized) * multiplier
        return number / 1_000_000_000 if number >= 1_000_000_000 else number
    except (TypeError, ValueError):
        return None


def _context_length(model: dict[str, Any]) -> int | None:
    for key in ("context_length", "max_context_length"):
        value = _number(model.get(key))
        if value is not None:
            return int(value)
    return None


def _model_validation(
    model_id: str, catalog: dict[str, dict[str, Any]], limit: Decimal
) -> dict[str, Any]:
    model = catalog.get(model_id)
    if model is None:
        return {
            "id": model_id, 
            "status": "unavailable",
            "within_cost_limit": False,
            "reason": "model is not present in the OpenRouter catalog"
            }
    
    pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}
    prompt_per_token = _money(pricing.get("prompt"))
    completion_per_token = _money(pricing.get("completion"))
    prompt_per_milion = prompt_per_token * 1_000_000 if prompt_per_token is not None else None
    completion_per_milion = completion_per_token * 1_000_000 if completion_per_token is not None else None
    known_costs = [cost for cost in (prompt_per_milion, completion_per_milion) if cost is not None]
    within_limit = bool(known_costs) and all(cost <= limit for cost in known_costs)

    context_length = _context_length(model)
    metadata_error = None if context_length is not None else "OpenRouter did not provide the model context length"
    return {
        "id": model_id,
        "status": "valid" if within_limit and metadata_error is None else (
            "cost-exceeded" if not within_limit else "metadata-unavailable"
        ),
        "within_cost_limit": within_limit,
        "cost_usd_per_million_tokens":{
            "prompt": str(prompt_per_milion) if prompt_per_milion is not None else None,
            "completion": str(completion_per_milion) if completion_per_milion is not None else None
        },
        "raw_pricing": pricing,
        "context_length": context_length,
        **({"reason": metadata_error} if metadata_error else {}),
    }


def select_models_for_mode(
    validations: list[dict[str, Any]],
    mode: str,
    *,
    count: int = FINAL_MODEL_COUNT,
    preferred_model_ids: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    profile = MODEL_PROFILES.get(mode)
    if profile is None:
        raise OpenRouterError(f"❌ Unknown model mode: {mode}. Use low, medium or high.")

    compatible = []
    for item in validations:
        context = item.get("context_length")
        if item.get("status") != "valid" or context is None:
            continue
        if context < profile["min_context"]:
            continue
        if profile["max_context"] is not None and context > profile["max_context"]:
            continue
        compatible.append(item)

    if len(compatible) < count:
        raise OpenRouterError(
            f"❌ Mode {mode} has only {len(compatible)} comparable valid models; "
            f"{count} are required. Check model size and context metadata."
        )

    compatible_by_id = {item["id"]: item for item in compatible}
    preferred = [
        compatible_by_id[model_id]
        for model_id in preferred_model_ids
        if model_id in compatible_by_id
    ][:count]
    preferred_ids = {item["id"] for item in preferred}
    remaining = [item for item in compatible if item["id"] not in preferred_ids]
    return preferred + random.SystemRandom().sample(remaining, count - len(preferred))



def _write_validation_cache(payload: dict[str, Any], output_path: Path | None = None) -> Path:
    path = output_path or (config_dir() / VALIDATION_CACHE_NAME)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return path


def validate_openrouter_setup(
    models: tuple[str, ...], *, output_path: Path | None = None
) -> Path:
    api_key = get_openrouter_api_key()
    if not api_key:
        raise OpenRouterError("‼️ OpenRouter API key is missing.")

    key_payload = _connect_with_retries(api_key)
    key_data = key_payload.get("data") if isinstance(key_payload.get("data"), dict) else {}
    limit_remainig = _money(key_data.get("limit_remaining"))
    has_funds = limit_remainig is None or limit_remainig > 0

    models_payload = _get_json("/models/user", api_key)
    model_items = models_payload.get("data")
    if not isinstance(model_items, list):
        raise OpenRouterError("‼️ OpenRouter returned an invalid model catalog")

    catalog = {
        item["id"]: item for item in model_items if isinstance(item, dict) and isinstance(item.get("id"), str)
    }

    max_cost = MAX_COST_PER_MILLION
    validations = [_model_validation(model, catalog, max_cost) for model in models]
    valid_models = [item for item in validations if item["status"] == "valid"]
    invalid_models = [item for item in validations if item["status"] != "valid"]
    checked_at = datetime.now(timezone.utc).isoformat()
    cache = {
        "status": "valid" if has_funds and len(valid_models) >= MIN_VALID_MODELS else "invalid",
        "last_checked_at": checked_at,
        "cost_limit_per_million_tokens": str(max_cost),
        "key": {
            "status": "valid" if has_funds else "no-funds",
            "label": key_data.get("label"),
            "limit": key_data.get("limit"),
            "limit_remaining": key_data.get("limit_remaining"),
            "usage": key_data.get("usage"),
            "expires_at": key_data.get("expires_at")
        },
        "models": validations,
        "results": valid_models,
        "errors": invalid_models
    }
    cache_path = _write_validation_cache(cache, output_path)

    if not has_funds:
        raise OpenRouterError(f"🛑 OpenRouter API key has no spending limit remaining. Details {cache_path}")

    invalid = [item["id"] for item in invalid_models]
    if len(valid_models) < MIN_VALID_MODELS:
        raise OpenRouterError(
            f"❌ At least {MIN_VALID_MODELS} OpenRouter models must be valid; "
            f"got {len(valid_models)}. Invalid models: {', '.join(invalid)}. Details: {cache_path}"
        )
    if invalid:
        log.warning("⚠️ Ignoring %s invalid OpenRouter models; continuing with %s valid models: %s", len(invalid), len(valid_models), ", ".join(invalid),)
    log.info("✅ Validated %s OpenRouter models; details saved to %s", len(validations), cache_path)

    return cache_path
                                                     