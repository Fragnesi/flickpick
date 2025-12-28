# flickpick Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI movie recommendation system that tracks ratings and gives personalized suggestions via content similarity and LLM-powered mood queries.

**Architecture:** Python CLI using Typer for commands, SQLite for local storage, TMDb API for movie data, and Ollama for natural language mood queries. Content-based recommendations via TF-IDF cosine similarity.

**Tech Stack:** Python 3.11+, Typer, Rich, httpx, scikit-learn, Ollama

**Design Doc:** `docs/plans/2025-12-28-flickpick-design.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `flickpick/__init__.py`
- Create: `flickpick/cli.py`
- Create: `tests/__init__.py`
- Create: `tests/test_cli.py`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "flickpick"
version = "0.1.0"
description = "CLI movie recommendation system"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "httpx>=0.25.0",
    "scikit-learn>=1.3.0",
    "ollama>=0.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[project.scripts]
flickpick = "flickpick.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
venv/
.env
*.db
```

**Step 3: Create flickpick/__init__.py**

```python
"""flickpick - CLI movie recommendation system."""

__version__ = "0.1.0"
```

**Step 4: Create minimal flickpick/cli.py**

```python
"""CLI entry point."""

import typer

app = typer.Typer(
    name="flickpick",
    help="Movie recommendation system - track ratings and get personalized suggestions.",
    no_args_is_help=True,
)


@app.command()
def version():
    """Show version."""
    from flickpick import __version__
    from rich import print
    print(f"[bold]flickpick[/bold] v{__version__}")


if __name__ == "__main__":
    app()
```

**Step 5: Create tests/__init__.py**

```python
"""Tests for flickpick."""
```

**Step 6: Create tests/test_cli.py**

```python
"""Test CLI commands."""

from typer.testing import CliRunner
from flickpick.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "flickpick" in result.stdout
    assert "0.1.0" in result.stdout
```

**Step 7: Create virtual environment and install**

Run:
```bash
cd /Users/fragnesi/repo/flickpick
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 8: Run tests**

Run: `pytest -v`
Expected: 1 passed

**Step 9: Verify CLI works**

Run: `flickpick version`
Expected: `flickpick v0.1.0`

**Step 10: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with Typer CLI"
```

---

## Task 2: Config Module

**Files:**
- Create: `flickpick/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing tests for config**

Create `tests/test_config.py`:

```python
"""Test configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def test_get_config_dir_creates_directory(tmp_path):
    """Config dir is created if it doesn't exist."""
    from flickpick.config import get_config_dir
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        config_dir = get_config_dir()
        assert config_dir.exists()
        assert config_dir == tmp_path / "flickpick"


def test_get_data_dir_creates_directory(tmp_path):
    """Data dir is created if it doesn't exist."""
    from flickpick.config import get_data_dir
    
    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
        data_dir = get_data_dir()
        assert data_dir.exists()
        assert data_dir == tmp_path / "flickpick"


def test_save_and_load_config(tmp_path):
    """Config can be saved and loaded."""
    from flickpick.config import get_config_dir, save_config, load_config
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        save_config({"tmdb_api_key": "test123"})
        config = load_config()
        assert config["tmdb_api_key"] == "test123"


def test_load_config_returns_empty_if_no_file(tmp_path):
    """Load returns empty dict if no config file."""
    from flickpick.config import load_config
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        config = load_config()
        assert config == {}


def test_get_tmdb_api_key(tmp_path):
    """Get TMDb API key from config."""
    from flickpick.config import save_config, get_tmdb_api_key
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        save_config({"tmdb_api_key": "mykey123"})
        assert get_tmdb_api_key() == "mykey123"


def test_get_tmdb_api_key_returns_none_if_missing(tmp_path):
    """Returns None if TMDb key not configured."""
    from flickpick.config import get_tmdb_api_key
    
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        assert get_tmdb_api_key() is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement config.py**

Create `flickpick/config.py`:

```python
"""Configuration management for flickpick."""

import json
import os
from pathlib import Path
from typing import Any


def get_config_dir() -> Path:
    """Get the config directory, creating it if needed."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    
    config_dir = base / "flickpick"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_data_dir() -> Path:
    """Get the data directory, creating it if needed."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    
    data_dir = base / "flickpick"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _config_file() -> Path:
    """Get path to config file."""
    return get_config_dir() / "config.json"


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file."""
    existing = load_config()
    existing.update(config)
    with open(_config_file(), "w") as f:
        json.dump(existing, f, indent=2)


def load_config() -> dict[str, Any]:
    """Load configuration from file."""
    config_file = _config_file()
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        return json.load(f)


def get_tmdb_api_key() -> str | None:
    """Get TMDb API key from config."""
    return load_config().get("tmdb_api_key")


def get_db_path() -> Path:
    """Get path to SQLite database."""
    return get_data_dir() / "flickpick.db"
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All 6 tests pass

**Step 5: Commit**

```bash
git add flickpick/config.py tests/test_config.py
git commit -m "feat: add config management module"
```

---

## Task 3: Database Module

**Files:**
- Create: `flickpick/db.py`
- Create: `tests/test_db.py`

**Step 1: Write failing tests**

Create `tests/test_db.py`:

```python
"""Test database operations."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def test_db(tmp_path):
    """Provide a test database."""
    from flickpick.db import Database
    
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


def test_database_creates_tables(test_db):
    """Database creates required tables on init."""
    cursor = test_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert "movies" in tables
    assert "ratings" in tables


def test_cache_movie(test_db):
    """Can cache movie metadata."""
    test_db.cache_movie(
        tmdb_id=550,
        title="Fight Club",
        year=1999,
        genres=["Drama", "Thriller"],
        plot="An insomniac office worker...",
        poster_url="https://image.tmdb.org/t/p/w500/poster.jpg",
    )
    
    movie = test_db.get_movie(550)
    assert movie is not None
    assert movie["title"] == "Fight Club"
    assert movie["year"] == 1999
    assert movie["genres"] == ["Drama", "Thriller"]


def test_get_movie_returns_none_if_not_found(test_db):
    """Get movie returns None for unknown tmdb_id."""
    assert test_db.get_movie(99999) is None


def test_add_rating(test_db):
    """Can add a rating for a movie."""
    # First cache the movie
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999)
    
    # Add rating
    test_db.add_rating(tmdb_id=550, rating=9)
    
    ratings = test_db.get_all_ratings()
    assert len(ratings) == 1
    assert ratings[0]["tmdb_id"] == 550
    assert ratings[0]["rating"] == 9


def test_add_watched_without_rating(test_db):
    """Can mark movie as watched without rating."""
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999)
    test_db.add_rating(tmdb_id=550, rating=None)
    
    ratings = test_db.get_all_ratings()
    assert len(ratings) == 1
    assert ratings[0]["rating"] is None


def test_get_all_ratings_with_movie_info(test_db):
    """Get ratings includes movie information."""
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999, genres=["Drama"])
    test_db.add_rating(tmdb_id=550, rating=9)
    
    ratings = test_db.get_all_ratings()
    assert ratings[0]["title"] == "Fight Club"
    assert ratings[0]["year"] == 1999


def test_is_watched(test_db):
    """Can check if movie is watched."""
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999)
    
    assert not test_db.is_watched(550)
    test_db.add_rating(tmdb_id=550, rating=8)
    assert test_db.is_watched(550)


def test_get_rated_movies_for_recommendations(test_db):
    """Get only rated movies (not just watched) for recommendations."""
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999, genres=["Drama"])
    test_db.cache_movie(tmdb_id=680, title="Pulp Fiction", year=1994, genres=["Crime"])
    
    test_db.add_rating(tmdb_id=550, rating=9)
    test_db.add_rating(tmdb_id=680, rating=None)  # Watched, not rated
    
    rated = test_db.get_rated_movies()
    assert len(rated) == 1
    assert rated[0]["tmdb_id"] == 550
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_db.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement db.py**

Create `flickpick/db.py`:

```python
"""Database operations for flickpick."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class Database:
    """SQLite database wrapper for flickpick."""

    def __init__(self, db_path: Path | None = None):
        """Initialize database connection."""
        if db_path is None:
            from flickpick.config import get_db_path
            db_path = get_db_path()
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY,
                tmdb_id INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                year INTEGER,
                genres TEXT,
                plot TEXT,
                poster_url TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY,
                tmdb_id INTEGER NOT NULL,
                rating INTEGER,
                watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tmdb_id) REFERENCES movies(tmdb_id)
            );

            CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies(tmdb_id);
            CREATE INDEX IF NOT EXISTS idx_ratings_tmdb_id ON ratings(tmdb_id);
        """)
        self.conn.commit()

    def cache_movie(
        self,
        tmdb_id: int,
        title: str,
        year: int | None = None,
        genres: list[str] | None = None,
        plot: str | None = None,
        poster_url: str | None = None,
    ) -> None:
        """Cache movie metadata from TMDb."""
        genres_json = json.dumps(genres) if genres else None
        self.conn.execute(
            """
            INSERT INTO movies (tmdb_id, title, year, genres, plot, poster_url)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id) DO UPDATE SET
                title = excluded.title,
                year = excluded.year,
                genres = excluded.genres,
                plot = excluded.plot,
                poster_url = excluded.poster_url,
                cached_at = CURRENT_TIMESTAMP
            """,
            (tmdb_id, title, year, genres_json, plot, poster_url),
        )
        self.conn.commit()

    def get_movie(self, tmdb_id: int) -> dict[str, Any] | None:
        """Get cached movie by TMDb ID."""
        cursor = self.conn.execute(
            "SELECT * FROM movies WHERE tmdb_id = ?", (tmdb_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_movie(row)

    def _row_to_movie(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert database row to movie dict."""
        movie = dict(row)
        if movie.get("genres"):
            movie["genres"] = json.loads(movie["genres"])
        return movie

    def add_rating(self, tmdb_id: int, rating: int | None = None) -> None:
        """Add or update a rating for a movie."""
        # Remove existing rating if any
        self.conn.execute("DELETE FROM ratings WHERE tmdb_id = ?", (tmdb_id,))
        # Add new rating
        self.conn.execute(
            "INSERT INTO ratings (tmdb_id, rating) VALUES (?, ?)",
            (tmdb_id, rating),
        )
        self.conn.commit()

    def get_all_ratings(self) -> list[dict[str, Any]]:
        """Get all ratings with movie info."""
        cursor = self.conn.execute(
            """
            SELECT r.*, m.title, m.year, m.genres, m.plot, m.poster_url
            FROM ratings r
            JOIN movies m ON r.tmdb_id = m.tmdb_id
            ORDER BY r.watched_at DESC
            """
        )
        return [self._row_to_rating(row) for row in cursor.fetchall()]

    def _row_to_rating(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert rating row to dict."""
        rating = dict(row)
        if rating.get("genres"):
            rating["genres"] = json.loads(rating["genres"])
        return rating

    def is_watched(self, tmdb_id: int) -> bool:
        """Check if movie is in watch history."""
        cursor = self.conn.execute(
            "SELECT 1 FROM ratings WHERE tmdb_id = ?", (tmdb_id,)
        )
        return cursor.fetchone() is not None

    def get_rated_movies(self) -> list[dict[str, Any]]:
        """Get movies with actual ratings (not just watched)."""
        cursor = self.conn.execute(
            """
            SELECT r.*, m.title, m.year, m.genres, m.plot, m.poster_url
            FROM ratings r
            JOIN movies m ON r.tmdb_id = m.tmdb_id
            WHERE r.rating IS NOT NULL
            ORDER BY r.rating DESC
            """
        )
        return [self._row_to_rating(row) for row in cursor.fetchall()]

    def get_all_cached_movies(self) -> list[dict[str, Any]]:
        """Get all cached movies."""
        cursor = self.conn.execute("SELECT * FROM movies")
        return [self._row_to_movie(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
```

**Step 4: Run tests**

Run: `pytest tests/test_db.py -v`
Expected: All 9 tests pass

**Step 5: Commit**

```bash
git add flickpick/db.py tests/test_db.py
git commit -m "feat: add database module for movies and ratings"
```

---

## Task 4: TMDb API Client

**Files:**
- Create: `flickpick/tmdb.py`
- Create: `tests/test_tmdb.py`

**Step 1: Write failing tests**

Create `tests/test_tmdb.py`:

```python
"""Test TMDb API client."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import httpx


@pytest.fixture
def mock_tmdb_response():
    """Sample TMDb API response."""
    return {
        "results": [
            {
                "id": 550,
                "title": "Fight Club",
                "release_date": "1999-10-15",
                "genre_ids": [18, 53],
                "overview": "An insomniac office worker...",
                "poster_path": "/poster.jpg",
                "vote_average": 8.4,
            }
        ]
    }


@pytest.fixture
def mock_genre_response():
    """Sample genre list response."""
    return {
        "genres": [
            {"id": 18, "name": "Drama"},
            {"id": 53, "name": "Thriller"},
            {"id": 28, "name": "Action"},
        ]
    }


@pytest.fixture
def mock_movie_detail_response():
    """Sample movie detail response."""
    return {
        "id": 550,
        "title": "Fight Club",
        "release_date": "1999-10-15",
        "genres": [{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}],
        "overview": "An insomniac office worker...",
        "poster_path": "/poster.jpg",
        "vote_average": 8.4,
        "runtime": 139,
        "tagline": "Mischief. Mayhem. Soap.",
    }


def test_tmdb_client_requires_api_key():
    """Client raises error without API key."""
    from flickpick.tmdb import TMDbClient
    
    with patch("flickpick.tmdb.get_tmdb_api_key", return_value=None):
        with pytest.raises(ValueError, match="TMDb API key not configured"):
            TMDbClient()


@pytest.mark.asyncio
async def test_search_movies(mock_tmdb_response, mock_genre_response):
    """Can search for movies."""
    from flickpick.tmdb import TMDbClient
    
    with patch("flickpick.tmdb.get_tmdb_api_key", return_value="fake_key"):
        client = TMDbClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tmdb_response
        mock_response.raise_for_status = MagicMock()
        
        mock_genre_resp = MagicMock()
        mock_genre_resp.json.return_value = mock_genre_response
        mock_genre_resp.raise_for_status = MagicMock()
        
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            # First call for genres, second for search
            mock_get.side_effect = [mock_genre_resp, mock_response]
            
            results = await client.search("Fight Club")
            
            assert len(results) == 1
            assert results[0]["title"] == "Fight Club"
            assert results[0]["year"] == 1999
            assert results[0]["tmdb_id"] == 550


@pytest.mark.asyncio
async def test_get_movie_details(mock_movie_detail_response):
    """Can get detailed movie info."""
    from flickpick.tmdb import TMDbClient
    
    with patch("flickpick.tmdb.get_tmdb_api_key", return_value="fake_key"):
        client = TMDbClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = mock_movie_detail_response
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            movie = await client.get_movie(550)
            
            assert movie["title"] == "Fight Club"
            assert movie["genres"] == ["Drama", "Thriller"]
            assert movie["runtime"] == 139


def test_parse_year_from_release_date():
    """Correctly parses year from release date."""
    from flickpick.tmdb import _parse_year
    
    assert _parse_year("1999-10-15") == 1999
    assert _parse_year("2023-01-01") == 2023
    assert _parse_year("") is None
    assert _parse_year(None) is None
```

**Step 2: Install pytest-asyncio**

Run: `pip install pytest-asyncio`

Update `pyproject.toml` to add pytest-asyncio to dev dependencies and configure it:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_tmdb.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 4: Implement tmdb.py**

Create `flickpick/tmdb.py`:

```python
"""TMDb API client."""

import httpx
from typing import Any

from flickpick.config import get_tmdb_api_key


BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def _parse_year(release_date: str | None) -> int | None:
    """Parse year from release date string."""
    if not release_date:
        return None
    try:
        return int(release_date[:4])
    except (ValueError, IndexError):
        return None


class TMDbClient:
    """Client for TMDb API."""

    def __init__(self, api_key: str | None = None):
        """Initialize client with API key."""
        self.api_key = api_key or get_tmdb_api_key()
        if not self.api_key:
            raise ValueError(
                "TMDb API key not configured. Run 'flickpick setup' or set TMDB_API_KEY."
            )
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            params={"api_key": self.api_key},
            timeout=30.0,
        )
        self._genres: dict[int, str] | None = None

    async def _ensure_genres(self) -> dict[int, str]:
        """Fetch and cache genre mapping."""
        if self._genres is None:
            response = await self._client.get("/genre/movie/list")
            response.raise_for_status()
            data = response.json()
            self._genres = {g["id"]: g["name"] for g in data["genres"]}
        return self._genres

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search for movies by title."""
        genres = await self._ensure_genres()
        response = await self._client.get(
            "/search/movie",
            params={"query": query},
        )
        response.raise_for_status()
        data = response.json()
        
        return [
            {
                "tmdb_id": movie["id"],
                "title": movie["title"],
                "year": _parse_year(movie.get("release_date")),
                "genres": [genres.get(gid, "Unknown") for gid in movie.get("genre_ids", [])],
                "plot": movie.get("overview"),
                "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
                "rating": movie.get("vote_average"),
            }
            for movie in data.get("results", [])
        ]

    async def get_movie(self, tmdb_id: int) -> dict[str, Any]:
        """Get detailed movie information."""
        response = await self._client.get(f"/movie/{tmdb_id}")
        response.raise_for_status()
        movie = response.json()
        
        return {
            "tmdb_id": movie["id"],
            "title": movie["title"],
            "year": _parse_year(movie.get("release_date")),
            "genres": [g["name"] for g in movie.get("genres", [])],
            "plot": movie.get("overview"),
            "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
            "rating": movie.get("vote_average"),
            "runtime": movie.get("runtime"),
            "tagline": movie.get("tagline"),
        }

    async def get_popular(self, page: int = 1) -> list[dict[str, Any]]:
        """Get popular movies."""
        genres = await self._ensure_genres()
        response = await self._client.get(
            "/movie/popular",
            params={"page": page},
        )
        response.raise_for_status()
        data = response.json()
        
        return [
            {
                "tmdb_id": movie["id"],
                "title": movie["title"],
                "year": _parse_year(movie.get("release_date")),
                "genres": [genres.get(gid, "Unknown") for gid in movie.get("genre_ids", [])],
                "plot": movie.get("overview"),
                "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
                "rating": movie.get("vote_average"),
            }
            for movie in data.get("results", [])
        ]

    async def discover(
        self,
        genres: list[int] | None = None,
        year: int | None = None,
        min_rating: float | None = None,
    ) -> list[dict[str, Any]]:
        """Discover movies with filters."""
        genre_map = await self._ensure_genres()
        params: dict[str, Any] = {"sort_by": "vote_average.desc", "vote_count.gte": 100}
        if genres:
            params["with_genres"] = ",".join(str(g) for g in genres)
        if year:
            params["year"] = year
        if min_rating:
            params["vote_average.gte"] = min_rating
        
        response = await self._client.get("/discover/movie", params=params)
        response.raise_for_status()
        data = response.json()
        
        return [
            {
                "tmdb_id": movie["id"],
                "title": movie["title"],
                "year": _parse_year(movie.get("release_date")),
                "genres": [genre_map.get(gid, "Unknown") for gid in movie.get("genre_ids", [])],
                "plot": movie.get("overview"),
                "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
                "rating": movie.get("vote_average"),
            }
            for movie in data.get("results", [])
        ]

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
```

**Step 5: Run tests**

Run: `pytest tests/test_tmdb.py -v`
Expected: All 4 tests pass

**Step 6: Commit**

```bash
git add flickpick/tmdb.py tests/test_tmdb.py pyproject.toml
git commit -m "feat: add TMDb API client"
```

---

## Task 5: Recommender Module

**Files:**
- Create: `flickpick/recommender.py`
- Create: `tests/test_recommender.py`

**Step 1: Write failing tests**

Create `tests/test_recommender.py`:

```python
"""Test recommendation engine."""

import pytest


def test_build_taste_profile():
    """Build taste profile from rated movies."""
    from flickpick.recommender import build_taste_profile
    
    rated_movies = [
        {"tmdb_id": 1, "rating": 9, "genres": ["Action", "Sci-Fi"], "plot": "A hero saves the world"},
        {"tmdb_id": 2, "rating": 8, "genres": ["Action", "Thriller"], "plot": "Explosive action"},
        {"tmdb_id": 3, "rating": 5, "genres": ["Comedy"], "plot": "Funny jokes"},
    ]
    
    profile = build_taste_profile(rated_movies)
    
    # Should favor highly-rated genres
    assert "Action" in profile["top_genres"]
    assert profile["genre_scores"]["Action"] > profile["genre_scores"]["Comedy"]


def test_find_similar_movies():
    """Find movies similar to given ones."""
    from flickpick.recommender import find_similar_movies
    
    source_movies = [
        {"tmdb_id": 1, "genres": ["Sci-Fi", "Action"], "plot": "Space battles and aliens"},
    ]
    
    candidates = [
        {"tmdb_id": 2, "genres": ["Sci-Fi", "Adventure"], "plot": "Space exploration journey"},
        {"tmdb_id": 3, "genres": ["Comedy", "Romance"], "plot": "Funny love story"},
        {"tmdb_id": 4, "genres": ["Sci-Fi", "Action"], "plot": "Alien invasion war"},
    ]
    
    similar = find_similar_movies(source_movies, candidates, limit=2)
    
    # Should return sci-fi movies, not comedy
    assert len(similar) == 2
    tmdb_ids = [m["tmdb_id"] for m in similar]
    assert 3 not in tmdb_ids  # Comedy shouldn't be in top 2


def test_score_by_taste_profile():
    """Score candidates against taste profile."""
    from flickpick.recommender import build_taste_profile, score_by_profile
    
    rated_movies = [
        {"tmdb_id": 1, "rating": 10, "genres": ["Horror", "Thriller"], "plot": "Scary stuff"},
        {"tmdb_id": 2, "rating": 9, "genres": ["Horror"], "plot": "More scares"},
    ]
    
    profile = build_taste_profile(rated_movies)
    
    candidates = [
        {"tmdb_id": 3, "genres": ["Horror", "Mystery"], "plot": "Dark mystery"},
        {"tmdb_id": 4, "genres": ["Comedy", "Family"], "plot": "Fun for everyone"},
    ]
    
    scored = score_by_profile(candidates, profile)
    
    # Horror movie should score higher
    assert scored[0]["tmdb_id"] == 3


def test_empty_ratings_returns_empty_profile():
    """Empty ratings produce empty profile."""
    from flickpick.recommender import build_taste_profile
    
    profile = build_taste_profile([])
    
    assert profile["top_genres"] == []
    assert profile["genre_scores"] == {}
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recommender.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement recommender.py**

Create `flickpick/recommender.py`:

```python
"""Recommendation engine for flickpick."""

from collections import defaultdict
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_taste_profile(rated_movies: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a taste profile from user's rated movies.
    
    Weights genres by rating (higher rated = more weight).
    """
    if not rated_movies:
        return {"top_genres": [], "genre_scores": {}, "avg_rating": 0}
    
    genre_scores: dict[str, float] = defaultdict(float)
    genre_counts: dict[str, int] = defaultdict(int)
    total_rating = 0
    
    for movie in rated_movies:
        rating = movie.get("rating", 5)  # Default to neutral
        weight = rating / 10.0  # Normalize to 0-1
        
        for genre in movie.get("genres", []):
            genre_scores[genre] += weight * rating
            genre_counts[genre] += 1
        
        total_rating += rating
    
    # Normalize by count
    for genre in genre_scores:
        genre_scores[genre] /= genre_counts[genre]
    
    # Sort by score
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "top_genres": [g[0] for g in sorted_genres[:5]],
        "genre_scores": dict(genre_scores),
        "avg_rating": total_rating / len(rated_movies) if rated_movies else 0,
    }


def find_similar_movies(
    source_movies: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find movies similar to source movies using content-based filtering.
    
    Uses TF-IDF on genres + plot for similarity.
    """
    if not source_movies or not candidates:
        return []
    
    def movie_to_text(movie: dict[str, Any]) -> str:
        """Convert movie to searchable text."""
        genres = " ".join(movie.get("genres", []))
        plot = movie.get("plot", "") or ""
        # Repeat genres to give them more weight
        return f"{genres} {genres} {genres} {plot}"
    
    # Combine source movies into single profile
    source_text = " ".join(movie_to_text(m) for m in source_movies)
    candidate_texts = [movie_to_text(m) for m in candidates]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = [source_text] + candidate_texts
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        # Empty vocabulary (no valid words)
        return candidates[:limit]
    
    # Calculate similarity (source is index 0)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Sort candidates by similarity
    scored_candidates = list(zip(candidates, similarities))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [
        {**movie, "similarity_score": float(score)}
        for movie, score in scored_candidates[:limit]
    ]


def score_by_profile(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score candidate movies against taste profile.
    
    Higher scores for movies matching preferred genres.
    """
    if not profile.get("genre_scores"):
        return candidates
    
    genre_scores = profile["genre_scores"]
    scored = []
    
    for movie in candidates:
        score = 0.0
        movie_genres = movie.get("genres", [])
        
        for genre in movie_genres:
            score += genre_scores.get(genre, 0)
        
        # Normalize by number of genres
        if movie_genres:
            score /= len(movie_genres)
        
        scored.append({**movie, "profile_score": score})
    
    # Sort by score descending
    scored.sort(key=lambda x: x["profile_score"], reverse=True)
    return scored
```

**Step 4: Run tests**

Run: `pytest tests/test_recommender.py -v`
Expected: All 4 tests pass

**Step 5: Commit**

```bash
git add flickpick/recommender.py tests/test_recommender.py
git commit -m "feat: add content-based recommendation engine"
```

---

## Task 6: LLM Integration (Ollama)

**Files:**
- Create: `flickpick/llm.py`
- Create: `tests/test_llm.py`

**Step 1: Write failing tests**

Create `tests/test_llm.py`:

```python
"""Test LLM integration."""

import pytest
from unittest.mock import patch, MagicMock


def test_build_mood_prompt():
    """Build prompt for mood-based recommendations."""
    from flickpick.llm import build_mood_prompt
    
    taste_profile = {
        "top_genres": ["Sci-Fi", "Thriller"],
        "avg_rating": 8.0,
    }
    
    prompt = build_mood_prompt(
        mood_query="something tense but not too scary",
        taste_profile=taste_profile,
    )
    
    assert "tense but not too scary" in prompt
    assert "Sci-Fi" in prompt
    assert "Thriller" in prompt


def test_parse_movie_suggestions():
    """Parse movie titles from LLM response."""
    from flickpick.llm import parse_movie_suggestions
    
    response = """Based on your preferences, here are some recommendations:

1. Arrival (2016) - A linguist works with the military to communicate with aliens.
2. Ex Machina (2014) - A programmer evaluates an AI.
3. Blade Runner 2049 (2017) - A blade runner discovers a secret.
"""
    
    movies = parse_movie_suggestions(response)
    
    assert len(movies) >= 3
    assert any("Arrival" in m for m in movies)
    assert any("Ex Machina" in m for m in movies)


def test_parse_handles_various_formats():
    """Parse handles different LLM output formats."""
    from flickpick.llm import parse_movie_suggestions
    
    # Format with dashes
    response1 = """
- The Matrix
- Inception
- Interstellar
"""
    assert len(parse_movie_suggestions(response1)) >= 3
    
    # Format with quotes
    response2 = '''
"Blade Runner"
"2001: A Space Odyssey"
'''
    assert len(parse_movie_suggestions(response2)) >= 2


@pytest.mark.asyncio
async def test_get_mood_recommendations_calls_ollama():
    """Mood recommendations call Ollama with correct prompt."""
    from flickpick.llm import get_mood_recommendations
    
    mock_response = {
        "message": {
            "content": "1. Inception (2010)\n2. The Matrix (1999)"
        }
    }
    
    with patch("flickpick.llm.ollama") as mock_ollama:
        mock_ollama.chat.return_value = mock_response
        
        result = await get_mood_recommendations(
            mood_query="mind-bending",
            taste_profile={"top_genres": ["Sci-Fi"]},
        )
        
        mock_ollama.chat.assert_called_once()
        assert "Inception" in result or "Matrix" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement llm.py**

Create `flickpick/llm.py`:

```python
"""LLM integration for mood-based recommendations."""

import re
from typing import Any

import ollama


DEFAULT_MODEL = "llama3.2"


def build_mood_prompt(
    mood_query: str,
    taste_profile: dict[str, Any],
    num_suggestions: int = 5,
) -> str:
    """Build a prompt for mood-based movie recommendations."""
    genres = ", ".join(taste_profile.get("top_genres", [])) or "various genres"
    
    return f"""You are a movie recommendation expert. Based on the user's mood and preferences, suggest {num_suggestions} movies.

User's favorite genres: {genres}
User's mood/request: "{mood_query}"

Rules:
1. Suggest exactly {num_suggestions} movies
2. Each movie should match the mood described
3. Consider the user's genre preferences but don't be limited by them
4. Format: numbered list with movie title and year
5. Include a brief one-line description for each

Respond with just the movie list, no preamble."""


def parse_movie_suggestions(response: str) -> list[str]:
    """Parse movie titles from LLM response.
    
    Handles various formats:
    - "1. Movie Title (Year)"
    - "- Movie Title"
    - '"Movie Title"'
    """
    movies = []
    lines = response.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering (1., 2., etc.)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Remove bullet points
        line = re.sub(r"^[-•*]\s*", "", line)
        # Remove quotes
        line = line.strip('"\'')
        
        if not line:
            continue
        
        # Extract title (everything before " - " description or first parenthesis content for year)
        # Match pattern: "Title (Year) - description" or "Title (Year)" or just "Title"
        match = re.match(r"^(.+?(?:\s*\(\d{4}\))?)", line)
        if match:
            title = match.group(1).strip()
            if title and len(title) > 1:
                movies.append(title)
    
    return movies


async def get_mood_recommendations(
    mood_query: str,
    taste_profile: dict[str, Any],
    model: str = DEFAULT_MODEL,
    num_suggestions: int = 5,
) -> str:
    """Get movie recommendations based on mood using Ollama.
    
    Returns the raw LLM response for display.
    """
    prompt = build_mood_prompt(mood_query, taste_profile, num_suggestions)
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response["message"]["content"]


def check_ollama_available(model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    """Check if Ollama is available and model is pulled.
    
    Returns (available, message).
    """
    try:
        # Check if ollama is running
        models = ollama.list()
        model_names = [m["name"].split(":")[0] for m in models.get("models", [])]
        
        if model.split(":")[0] not in model_names:
            return False, f"Model '{model}' not found. Run: ollama pull {model}"
        
        return True, "Ollama ready"
    except Exception as e:
        if "refused" in str(e).lower() or "connect" in str(e).lower():
            return False, "Ollama not running. Start with: ollama serve"
        return False, f"Ollama error: {e}"
```

**Step 4: Run tests**

Run: `pytest tests/test_llm.py -v`
Expected: All 4 tests pass

**Step 5: Commit**

```bash
git add flickpick/llm.py tests/test_llm.py
git commit -m "feat: add Ollama LLM integration for mood queries"
```

---

## Task 7: CLI Commands - Core

**Files:**
- Modify: `flickpick/cli.py`
- Update: `tests/test_cli.py`

**Step 1: Write failing tests for search command**

Update `tests/test_cli.py`:

```python
"""Test CLI commands."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from typer.testing import CliRunner
from flickpick.cli import app

runner = CliRunner()


def test_version():
    """Version command shows version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "flickpick" in result.stdout
    assert "0.1.0" in result.stdout


def test_search_requires_query():
    """Search requires a query argument."""
    with patch("flickpick.cli.get_tmdb_api_key", return_value="fake"):
        result = runner.invoke(app, ["search"])
        assert result.exit_code != 0


def test_search_shows_results():
    """Search displays movie results."""
    mock_results = [
        {
            "tmdb_id": 550,
            "title": "Fight Club",
            "year": 1999,
            "genres": ["Drama", "Thriller"],
            "rating": 8.4,
        }
    ]
    
    with patch("flickpick.cli.get_tmdb_api_key", return_value="fake"):
        with patch("flickpick.cli.TMDbClient") as MockClient:
            instance = MockClient.return_value
            instance.search = AsyncMock(return_value=mock_results)
            instance.close = AsyncMock()
            
            result = runner.invoke(app, ["search", "fight club"])
            
            assert result.exit_code == 0
            assert "Fight Club" in result.stdout
            assert "1999" in result.stdout


def test_rate_saves_rating(tmp_path):
    """Rate command saves rating to database."""
    mock_movie = {
        "tmdb_id": 550,
        "title": "Fight Club",
        "year": 1999,
        "genres": ["Drama"],
        "plot": "Test plot",
        "poster_url": None,
    }
    mock_search = [mock_movie]
    
    with patch("flickpick.cli.get_tmdb_api_key", return_value="fake"):
        with patch("flickpick.cli.TMDbClient") as MockClient:
            instance = MockClient.return_value
            instance.search = AsyncMock(return_value=mock_search)
            instance.get_movie = AsyncMock(return_value=mock_movie)
            instance.close = AsyncMock()
            
            with patch("flickpick.cli.Database") as MockDB:
                db_instance = MockDB.return_value
                db_instance.is_watched.return_value = False
                
                result = runner.invoke(app, ["rate", "Fight Club", "9"])
                
                assert result.exit_code == 0
                db_instance.cache_movie.assert_called_once()
                db_instance.add_rating.assert_called_once()


def test_history_shows_watched_movies(tmp_path):
    """History command displays watch history."""
    mock_ratings = [
        {
            "tmdb_id": 550,
            "title": "Fight Club",
            "year": 1999,
            "rating": 9,
            "watched_at": "2024-01-15",
            "genres": ["Drama"],
        }
    ]
    
    with patch("flickpick.cli.Database") as MockDB:
        db_instance = MockDB.return_value
        db_instance.get_all_ratings.return_value = mock_ratings
        
        result = runner.invoke(app, ["history"])
        
        assert result.exit_code == 0
        assert "Fight Club" in result.stdout
```

**Step 2: Implement full cli.py**

Update `flickpick/cli.py`:

```python
"""CLI entry point for flickpick."""

import asyncio
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

from flickpick import __version__
from flickpick.config import get_tmdb_api_key, save_config
from flickpick.db import Database
from flickpick.tmdb import TMDbClient
from flickpick.recommender import build_taste_profile, find_similar_movies, score_by_profile
from flickpick.llm import get_mood_recommendations, check_ollama_available

app = typer.Typer(
    name="flickpick",
    help="Movie recommendation system - track ratings and get personalized suggestions.",
    no_args_is_help=True,
)
console = Console()


def run_async(coro):
    """Run async function in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)


@app.command()
def version():
    """Show version."""
    print(f"[bold]flickpick[/bold] v{__version__}")


@app.command()
def setup():
    """Interactive setup wizard."""
    console.print(Panel.fit("[bold]Welcome to flickpick![/bold]\nLet's get you set up.", border_style="blue"))
    
    # TMDb API Key
    console.print("\n[bold]1. TMDb API Key[/bold]")
    console.print("   Get one free at: [link]https://www.themoviedb.org/settings/api[/link]")
    
    api_key = Prompt.ask("   Paste your API key")
    if api_key:
        save_config({"tmdb_api_key": api_key.strip()})
        console.print("   [green]✓[/green] API key saved!")
    
    # Ollama check
    console.print("\n[bold]2. Ollama (for mood-based recommendations)[/bold]")
    available, message = check_ollama_available()
    if available:
        console.print(f"   [green]✓[/green] {message}")
    else:
        console.print(f"   [yellow]![/yellow] {message}")
        console.print("   Install: [code]brew install ollama && ollama pull llama3.2[/code]")
    
    console.print("\n[green]✓ Setup complete![/green]")
    console.print("Try: [code]flickpick search inception[/code]")


@app.command()
def search(query: str = typer.Argument(..., help="Movie title to search for")):
    """Search for movies on TMDb."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def do_search():
        client = TMDbClient(api_key)
        try:
            results = await client.search(query)
            return results
        finally:
            await client.close()
    
    results = run_async(do_search())
    
    if not results:
        console.print(f"No movies found for '{query}'")
        return
    
    table = Table(title=f"Search results for '{query}'")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("Rating", width=6)
    
    for i, movie in enumerate(results[:10], 1):
        genres = ", ".join(movie.get("genres", [])[:3])
        rating = f"{movie.get('rating', 0):.1f}" if movie.get("rating") else "-"
        table.add_row(
            str(i),
            movie["title"],
            str(movie.get("year", "-")),
            genres,
            rating,
        )
    
    console.print(table)


@app.command()
def info(title: str = typer.Argument(..., help="Movie title")):
    """Show detailed movie information."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def get_info():
        client = TMDbClient(api_key)
        try:
            results = await client.search(title)
            if not results:
                return None
            movie = await client.get_movie(results[0]["tmdb_id"])
            return movie
        finally:
            await client.close()
    
    movie = run_async(get_info())
    
    if not movie:
        console.print(f"Movie not found: '{title}'")
        return
    
    # Display movie info
    console.print(Panel.fit(
        f"[bold]{movie['title']}[/bold] ({movie.get('year', 'N/A')})\n\n"
        f"[dim]{movie.get('tagline', '')}[/dim]\n\n"
        f"[bold]Genres:[/bold] {', '.join(movie.get('genres', []))}\n"
        f"[bold]Runtime:[/bold] {movie.get('runtime', 'N/A')} min\n"
        f"[bold]Rating:[/bold] {movie.get('rating', 'N/A')}/10\n\n"
        f"[bold]Plot:[/bold]\n{movie.get('plot', 'No plot available.')}",
        border_style="blue",
    ))


@app.command()
def rate(
    title: str = typer.Argument(..., help="Movie title"),
    rating: int = typer.Argument(..., help="Rating from 1-10"),
):
    """Rate a movie (1-10)."""
    if not 1 <= rating <= 10:
        console.print("[red]Error:[/red] Rating must be between 1 and 10")
        raise typer.Exit(1)
    
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def find_and_rate():
        client = TMDbClient(api_key)
        try:
            results = await client.search(title)
            if not results:
                return None
            movie = await client.get_movie(results[0]["tmdb_id"])
            return movie
        finally:
            await client.close()
    
    movie = run_async(find_and_rate())
    
    if not movie:
        console.print(f"Movie not found: '{title}'")
        raise typer.Exit(1)
    
    db = Database()
    db.cache_movie(
        tmdb_id=movie["tmdb_id"],
        title=movie["title"],
        year=movie.get("year"),
        genres=movie.get("genres"),
        plot=movie.get("plot"),
        poster_url=movie.get("poster_url"),
    )
    db.add_rating(movie["tmdb_id"], rating)
    
    console.print(f"[green]✓[/green] Rated [bold]{movie['title']}[/bold] ({movie.get('year', 'N/A')}) - {rating}/10")


@app.command()
def watched(title: str = typer.Argument(..., help="Movie title")):
    """Mark a movie as watched (without rating)."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def find_movie():
        client = TMDbClient(api_key)
        try:
            results = await client.search(title)
            if not results:
                return None
            movie = await client.get_movie(results[0]["tmdb_id"])
            return movie
        finally:
            await client.close()
    
    movie = run_async(find_movie())
    
    if not movie:
        console.print(f"Movie not found: '{title}'")
        raise typer.Exit(1)
    
    db = Database()
    db.cache_movie(
        tmdb_id=movie["tmdb_id"],
        title=movie["title"],
        year=movie.get("year"),
        genres=movie.get("genres"),
        plot=movie.get("plot"),
        poster_url=movie.get("poster_url"),
    )
    db.add_rating(movie["tmdb_id"], rating=None)
    
    console.print(f"[green]✓[/green] Marked [bold]{movie['title']}[/bold] ({movie.get('year', 'N/A')}) as watched")


@app.command()
def history():
    """Show your watch history."""
    db = Database()
    ratings = db.get_all_ratings()
    
    if not ratings:
        console.print("No movies in your history yet. Try: [code]flickpick rate 'Inception' 9[/code]")
        return
    
    table = Table(title="Your Watch History")
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Rating", width=8)
    table.add_column("Genres")
    
    for entry in ratings:
        rating_str = f"{entry['rating']}/10" if entry.get("rating") else "[dim]watched[/dim]"
        genres = ", ".join(entry.get("genres", [])[:3]) if entry.get("genres") else "-"
        table.add_row(
            entry["title"],
            str(entry.get("year", "-")),
            rating_str,
            genres,
        )
    
    console.print(table)


@app.command(name="like")
def like_movies(
    titles: list[str] = typer.Argument(..., help="Movie title(s) to find similar movies for"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recommendations"),
):
    """Find movies similar to the ones you specify."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def get_similar():
        client = TMDbClient(api_key)
        try:
            # Get source movies
            source_movies = []
            for title in titles:
                results = await client.search(title)
                if results:
                    movie = await client.get_movie(results[0]["tmdb_id"])
                    source_movies.append(movie)
            
            if not source_movies:
                return None, []
            
            # Get candidates from popular + discover
            candidates = await client.get_popular()
            
            # Get more candidates based on source genres
            all_genres = set()
            for m in source_movies:
                all_genres.update(m.get("genres", []))
            
            return source_movies, candidates
        finally:
            await client.close()
    
    source_movies, candidates = run_async(get_similar())
    
    if not source_movies:
        console.print("Could not find any of the specified movies")
        raise typer.Exit(1)
    
    # Filter out source movies from candidates
    source_ids = {m["tmdb_id"] for m in source_movies}
    candidates = [c for c in candidates if c["tmdb_id"] not in source_ids]
    
    # Also filter out already watched
    db = Database()
    watched_ids = {r["tmdb_id"] for r in db.get_all_ratings()}
    candidates = [c for c in candidates if c["tmdb_id"] not in watched_ids]
    
    similar = find_similar_movies(source_movies, candidates, limit=limit)
    
    source_titles = ", ".join(m["title"] for m in source_movies)
    console.print(f"\n[bold]Movies similar to:[/bold] {source_titles}\n")
    
    table = Table()
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("Match", width=6)
    
    for i, movie in enumerate(similar, 1):
        genres = ", ".join(movie.get("genres", [])[:3])
        score = movie.get("similarity_score", 0)
        match_pct = f"{score * 100:.0f}%"
        table.add_row(
            str(i),
            movie["title"],
            str(movie.get("year", "-")),
            genres,
            match_pct,
        )
    
    console.print(table)


@app.command()
def suggest(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recommendations"),
):
    """Get personalized recommendations based on your ratings."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    db = Database()
    rated_movies = db.get_rated_movies()
    
    if len(rated_movies) < 3:
        console.print("Rate at least 3 movies first for personalized recommendations.")
        console.print("Try: [code]flickpick rate 'Movie Name' 8[/code]")
        return
    
    profile = build_taste_profile(rated_movies)
    
    async def get_candidates():
        client = TMDbClient(api_key)
        try:
            return await client.get_popular()
        finally:
            await client.close()
    
    candidates = run_async(get_candidates())
    
    # Filter out watched movies
    watched_ids = {r["tmdb_id"] for r in db.get_all_ratings()}
    candidates = [c for c in candidates if c["tmdb_id"] not in watched_ids]
    
    # Score by profile
    recommendations = score_by_profile(candidates, profile)[:limit]
    
    console.print(f"\n[bold]Your top genres:[/bold] {', '.join(profile['top_genres'])}\n")
    
    table = Table(title="Recommended for You")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("TMDb", width=6)
    
    for i, movie in enumerate(recommendations, 1):
        genres = ", ".join(movie.get("genres", [])[:3])
        rating = f"{movie.get('rating', 0):.1f}" if movie.get("rating") else "-"
        table.add_row(
            str(i),
            movie["title"],
            str(movie.get("year", "-")),
            genres,
            rating,
        )
    
    console.print(table)


@app.command()
def mood(
    query: str = typer.Argument(..., help="Describe what you're in the mood for"),
):
    """Get recommendations based on your mood (uses AI)."""
    # Check Ollama
    available, message = check_ollama_available()
    if not available:
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(1)
    
    db = Database()
    rated_movies = db.get_rated_movies()
    profile = build_taste_profile(rated_movies)
    
    with console.status("[bold blue]Thinking...[/bold blue]"):
        response = run_async(get_mood_recommendations(query, profile))
    
    console.print(Panel.fit(
        f"[bold]Mood:[/bold] {query}\n\n{response}",
        title="AI Recommendations",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
```

**Step 3: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All tests pass

**Step 4: Manual smoke test**

```bash
# Test help
flickpick --help

# Test version
flickpick version

# Test search (requires TMDb key)
flickpick setup  # Enter your TMDb key
flickpick search "inception"
```

**Step 5: Commit**

```bash
git add flickpick/cli.py tests/test_cli.py
git commit -m "feat: implement full CLI with all commands"
```

---

## Task 8: README and Final Polish

**Files:**
- Create: `README.md`

**Step 1: Create README.md**

```markdown
# flickpick

A CLI movie recommendation system that tracks your ratings and gives personalized suggestions.

## Features

- **Track movies** - Rate what you've watched, build your taste profile
- **"Movies like X"** - Find similar movies based on genres and plot
- **AI-powered mood search** - "something tense but not too violent" → personalized picks
- **Local-first** - All data stored locally, no account needed

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/flickpick.git
cd flickpick
pip install -e .
```

## Setup

```bash
flickpick setup
```

This will:
1. Ask for your [TMDb API key](https://www.themoviedb.org/settings/api) (free)
2. Check if Ollama is installed for AI features

### Install Ollama (optional, for mood search)

```bash
brew install ollama
ollama pull llama3.2
```

## Usage

### Track Movies

```bash
# Rate a movie (1-10)
flickpick rate "Inception" 9

# Mark as watched without rating
flickpick watched "The Matrix"

# View your history
flickpick history
```

### Get Recommendations

```bash
# Based on your taste profile
flickpick suggest

# Find similar movies
flickpick like "Blade Runner"
flickpick like "Inception" "Tenet" "Interstellar"

# Mood-based (requires Ollama)
flickpick mood "something tense but not too scary"
flickpick mood "feel-good comedy for a rainy day"
```

### Browse

```bash
# Search TMDb
flickpick search "batman"

# Get movie details
flickpick info "The Dark Knight"
```

## Data Storage

- Config: `~/.config/flickpick/config.json`
- Database: `~/.local/share/flickpick/flickpick.db`

## License

MIT
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## Task 9: Run Full Test Suite

**Step 1: Run all tests with coverage**

```bash
pytest -v --cov=flickpick --cov-report=term-missing
```

Expected: All tests pass, reasonable coverage

**Step 2: Fix any failing tests**

If any tests fail, fix them before proceeding.

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: finalize test suite"
```

---

## Summary

After completing all tasks, you will have:

1. **Project scaffolding** - pyproject.toml, CLI entry point
2. **Config module** - XDG-compliant paths, API key storage
3. **Database module** - SQLite for movies and ratings
4. **TMDb client** - Async API client for movie data
5. **Recommender** - TF-IDF content similarity + taste profiling
6. **LLM integration** - Ollama for mood-based queries
7. **Full CLI** - All commands: rate, watched, history, search, info, like, suggest, mood
8. **README** - Setup and usage documentation
9. **Test suite** - Unit tests for all modules

**Total estimated time:** 2-3 hours for an experienced developer
