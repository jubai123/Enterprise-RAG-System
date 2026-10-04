from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.config.models import AppConfig

ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::([^}]*))?\}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "main.yaml"


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    match = ENV_PATTERN.match(value)
    if not match:
        return value
    name, default = match.groups()
    return os.getenv(name, default or "")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file) if path.suffix.lower() == ".json" else yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be an object: {path}")
    return value


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """读取 main baseline，并在环境变量展开后一次性完成类型校验。"""
    load_dotenv(PROJECT_ROOT / ".env")
    raw = _read_config(MAIN_CONFIG_PATH)
    if config_path:
        override_path = Path(config_path).resolve()
        if override_path != MAIN_CONFIG_PATH.resolve():
            raw = _deep_merge(raw, _read_config(override_path))
    return AppConfig.model_validate(_expand_env(raw))
