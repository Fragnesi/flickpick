# AGENTS.md - Agent Instructions for flickpick

**Generated:** 2025-12-30
**Branch:** main

## OVERVIEW

CLI movie recommendation system using TMDb API + local Ollama LLM. Tracks ratings, builds taste profiles, suggests similar movies via TF-IDF/cosine similarity.

## STRUCTURE

```
flickpick/
├── flickpick/
│   ├── __init__.py     # Version only
│   ├── cli.py          # Typer commands (entry point)
│   ├── config.py       # XDG paths, API key storage
│   ├── db.py           # SQLite: movies cache + ratings
│   ├── llm.py          # Ollama integration for mood queries
│   ├── recommender.py  # TF-IDF similarity + taste profiling
│   └── tmdb.py         # Async TMDb API client (httpx)
├── tests/              # pytest, mirrors source 1:1
├── docs/plans/         # Design + implementation docs
└── pyproject.toml      # hatchling build, deps, pytest config
```

## COMMANDS

```bash
# Install (editable with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest

# Run single test file
pytest tests/test_cli.py
pytest tests/test_db.py -v

# Run single test function
pytest tests/test_cli.py::test_version
pytest tests/test_db.py::test_cache_movie -v

# Run tests with coverage
pytest --cov=flickpick

# Run CLI
flickpick --help
flickpick setup
flickpick search "inception"
flickpick rate "Inception" 9
```

## WHERE TO LOOK

| Task | Location | Key Functions/Classes |
|------|----------|----------------------|
| Add CLI command | `flickpick/cli.py` | `@app.command()` decorator |
| Change DB schema | `flickpick/db.py` | `Database._create_tables()` |
| TMDb API changes | `flickpick/tmdb.py` | `TMDbClient` (async methods) |
| Recommendation logic | `flickpick/recommender.py` | `find_similar_movies()`, `score_by_profile()` |
| LLM/mood features | `flickpick/llm.py` | `get_mood_recommendations()` |
| Config/paths | `flickpick/config.py` | `get_config_dir()`, `get_data_dir()` |

## CODE STYLE

### Imports
Order: stdlib -> third-party -> local. One blank line between groups.

### Type Hints
Use Python 3.11+ syntax (`list[str]`, `dict[str, Any]`, `str | None`). Never use `typing.List`, `typing.Dict`, `typing.Optional`.

### Naming
- Functions/variables: `snake_case`
- Classes: `PascalCase`  
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Error Handling
- CLI errors: Print with Rich, then `raise typer.Exit(1)`
- API errors: Let httpx errors propagate with `response.raise_for_status()`
- No bare `except:` - always specify exception type

### Async Pattern
TMDb client is async. CLI bridges with `run_async(coro)` wrapper using `asyncio.run()`.

### Database Pattern
- No ORM - direct `sqlite3`
- Instantiate `Database()` per-command, never use singletons
- Genres stored as JSON strings

## TESTING PATTERNS

### Structure
Tests mirror source files 1:1. Import inside test functions for easier mocking.

### Fixtures
Use `tmp_path` for database tests. Always close DB in fixture teardown:
```python
@pytest.fixture
def test_db(tmp_path):
    from flickpick.db import Database
    db = Database(tmp_path / "test.db")
    yield db
    db.close()
```

### Mocking
All external APIs (TMDb, Ollama) fully mocked. Use `AsyncMock` for async methods:
```python
with patch("flickpick.cli.TMDbClient") as MockClient:
    instance = MockClient.return_value
    instance.search = AsyncMock(return_value=mock_results)
    instance.close = AsyncMock()
```

### Async Tests
Use `@pytest.mark.asyncio` decorator. `asyncio_mode = "auto"` in pyproject.toml.

## ANTI-PATTERNS (DO NOT)

| Don't | Why |
|-------|-----|
| `# type: ignore`, `cast()`, `Any` escapes | Defeats type safety |
| Global `Database()` singleton | Creates test isolation issues |
| `await` in sync CLI commands | Use `run_async()` wrapper |
| Empty `except:` blocks | Hide real errors |

## KEY DEPENDENCIES

| Package | Purpose |
|---------|---------|
| `typer` | CLI framework with type hints |
| `rich` | Terminal output formatting (tables, panels) |
| `httpx` | Async HTTP client for TMDb API |
| `scikit-learn` | TF-IDF vectorization, cosine similarity |
| `ollama` | Local LLM integration |
| `pytest` | Testing framework |
| `pytest-asyncio` | Async test support |

## PROJECT-SPECIFIC NOTES

- **TMDb API key required**: Free at themoviedb.org/settings/api
- **Ollama optional**: Only for `mood` command; `brew install ollama && ollama pull llama3.2`
- **No CI/CD**: No GitHub Actions or pre-commit hooks configured
- **XDG paths**: Config at `~/.config/flickpick/`, data at `~/.local/share/flickpick/`
- **Genres as JSON**: `movies.genres` column stores JSON array string
- **TF-IDF weighting**: Genres repeated 3x in text for higher similarity weight
