from __future__ import annotations

import json
import logging
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
REQIEST_TIMEOUT_SECONDS = 20
ATTEMPT_DELAYS_SECONDS = (10, 20)

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


def _model_validation(model_id: str, catalog: dict[str, dict[str, Any]], limit: Decimal) -> dict[str, Any]:
    model = catalog.get(model_id)
    if model is None:
        return {
            "id": model_id, 
            "status": "unavaliable",
            "within_cost_limit": False,
            "reason": "model is not present in the OpenRouter catalog"
            }
    pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}
    prompt_per_token = _money(pricing.get("prompt"))
    completion_per_token = _money(pricing.get("completion"))
    prompt_per_milion = completion_per_token * 1_000_000 if prompt_per_token is not None else None
    completion_per_milion = completion_per_token * 1_000_000 if completion_per_token is not None else None
    known_costs = [cost for cost in (prompt_per_milion, completion_per_milion) if cost is not None]
    within_limit = bool(known_costs) and all(cost <= limit for cost in known_costs)


    return {
        "id": model_id,
        "status": "valid" if within_limit else "cost-exceeded",
        "within_cost_limit": within_limit,
        "cost_usd_per_million_tokens":{
            "prompt": str(prompt_per_milion) if prompt_per_milion is not None else None,
            "completion": str(completion_per_milion) if completion_per_milion is not None else None
        },
        "raw_pricing": pricing
    }



def _write_validation_cache(payload: dict[str, Any], output_path: Path | None = None) -> Path:
    path = output_path or (config_dir() / VALIDATION_CACHE_NAME)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return path


def validate_openrouter_setup(models: tuple[str, ...], *, output_path: Path | None = None) -> Path:
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
    checked_at = datetime.now(timezone.utc).isoformat()
    cache = {
        "status": "valid" if has_funds and all(item["status"] == "valid" for item in validations) else "invalid",
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
        "models": validations
    }
    cache_path = _write_validation_cache(cache, output_path)

    if not has_funds:
        raise OpenRouterError(f"🛑 OpenRouter API key has no spending limit remaining. Details {cache_path}")

    invalid = [item["id"] for item in validations if item["status"] != "valid"]
    if invalid:
        raise OpenRouterError(f"❌ OpenRouter model validation failed for: {", ".join(invalid)}, Details: {cache_path}")
    log.info("✅ Validated %s OpenRouter models; details saved to %s", len(validations), cache_path)

    return cache_path
                                                     