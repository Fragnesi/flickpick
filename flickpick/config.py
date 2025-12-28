import json
import os
from pathlib import Path
from typing import Any


def get_config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    
    config_dir = base / "flickpick"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    
    data_dir = base / "flickpick"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _config_file() -> Path:
    return get_config_dir() / "config.json"


def save_config(config: dict[str, Any]) -> None:
    existing = load_config()
    existing.update(config)
    with open(_config_file(), "w") as f:
        json.dump(existing, f, indent=2)


def load_config() -> dict[str, Any]:
    config_file = _config_file()
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        return json.load(f)


def get_tmdb_api_key() -> str | None:
    return load_config().get("tmdb_api_key")


def get_db_path() -> Path:
    return get_data_dir() / "flickpick.db"
