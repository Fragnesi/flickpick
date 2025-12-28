import os
from pathlib import Path
from unittest.mock import patch

import pytest


def test_get_config_dir_creates_directory(tmp_path):
    from flickpick.config import get_config_dir
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        config_dir = get_config_dir()
        assert config_dir.exists()
        assert config_dir == tmp_path / "flickpick"


def test_get_data_dir_creates_directory(tmp_path):
    from flickpick.config import get_data_dir
    
    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
        data_dir = get_data_dir()
        assert data_dir.exists()
        assert data_dir == tmp_path / "flickpick"


def test_save_and_load_config(tmp_path):
    from flickpick.config import get_config_dir, save_config, load_config
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        save_config({"tmdb_api_key": "test123"})
        config = load_config()
        assert config["tmdb_api_key"] == "test123"


def test_load_config_returns_empty_if_no_file(tmp_path):
    from flickpick.config import load_config
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        config = load_config()
        assert config == {}


def test_get_tmdb_api_key(tmp_path):
    from flickpick.config import save_config, get_tmdb_api_key
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        save_config({"tmdb_api_key": "mykey123"})
        assert get_tmdb_api_key() == "mykey123"


def test_get_tmdb_api_key_returns_none_if_missing(tmp_path):
    from flickpick.config import get_tmdb_api_key
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        assert get_tmdb_api_key() is None
