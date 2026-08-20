from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
from typing import Callable
from .config import DEFAULT_INITIAL_MODEL

CONFIG_DIR_ENV = "WORST_SHINKA_CONFIG_DIR"
API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_PREFIX = "sk-or-v1-"

def config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "worst-shinka"

def credentials_path() -> Path:
    return config_dir() / ".env"

def _saved_key() -> str | None:
    path = credentials_path()
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    prefix = f"{API_KEY_ENV}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix):].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"',"'"}:
                value = value[1:-1]
            return value.strip() or None

    return None

def api_key_source() -> str | None:
    if os.environ.get(API_KEY_ENV, "").strip():
        return f"environment variable {API_KEY_ENV}"
    if _saved_key():
        return str(credentials_path())
    return None

def get_openrouter_api_key() -> str | None:
    return os.environ.get(API_KEY_ENV, "").strip() or _saved_key()

def save_openrouter_api_key(api_key: str) -> Path:
    if "\n" in api_key or "\r" in api_key:
        raise ValueError("API key must be a single line")
    value = api_key.strip()
    if not value.startswith(OPENROUTER_PREFIX):
        raise ValueError(f"API key must start with: {OPENROUTER_PREFIX}")
    if len(value) <= len(OPENROUTER_PREFIX) + 40:
        raise ValueError("API key is incomplete")
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = credentials_path()
    path.write_text(f"{API_KEY_ENV}={value}\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return Path


def login(prompt: Callable[[str], str] | None = None) -> Path:
    read_secret = prompt or getpass.getpass
    key = read_secret("OpenRouter API key: ")
    return save_openrouter_api_key(key)

def logout() -> bool:
    path = credentials_path()
    if not path.exists():
        return False
    path.unlink()
    return True

def display_status() -> str:
    source = api_key_source()
    key = get_openrouter_api_key()
    lines = [
        f"User config directory: {config_dir()}",
        f"Default initial model: {DEFAULT_INITIAL_MODEL}", 
        "Default results directory: results",
        "Default generations: 10", 
        "Default workers: 1",
        "Default parents: 4",
        f"OpenRouter: {'connected' if source else 'not connected'}"
    ]
    if source:
        lines.append(f"Credential source: {source}")
    if key:
        lines.append(f"API key: {mask_api_key(key)}")
    return "\n".join(lines)

def mask_api_key(api_key: str) -> str:
    value = api_key.strip()
    if len(value) < 8:
        return "*" * len(value)

    return f"{value[:4]}{'*'*min(12, len(value)-8)}{value[-4:]}"

