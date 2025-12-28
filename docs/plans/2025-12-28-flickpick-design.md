# flickpick - Movie Recommendation System Design

**Date:** 2025-12-28  
**Status:** Approved

## Overview

A CLI tool that tracks your movie ratings and gives personalized recommendations using a hybrid approach: content similarity (genres, plot) + LLM-powered natural language queries.

### Goals
- Personal use + share recommendations with friends
- Functional tool that "just works"
- CLI-first, potential web UI later

### Non-Goals (v1)
- TV series support
- Web UI
- User accounts / multi-user
- Social features
- Watchlist / "want to watch"
- Streaming availability

---

## Core Features

### Commands

```bash
# Track movies
flickpick rate "Inception" 9          # Rate a movie 1-10
flickpick watched "The Matrix"        # Mark as watched (no rating)
flickpick history                      # Show your watch history

# Get recommendations
flickpick suggest                      # Based on your taste profile
flickpick like "Blade Runner"          # Find similar movies
flickpick like "Inception" "Tenet"     # Similar to multiple
flickpick mood "tense slow-burn thriller, not too violent"

# Browse
flickpick search "batman"              # Search TMDb
flickpick info "The Dark Knight"       # Show movie details
```

### Data Flow
1. User rates/watches movies → stored in local SQLite
2. Movie metadata fetched from TMDb (cached locally)
3. `suggest` analyzes ratings to find patterns
4. `mood` sends query + taste profile to local Ollama → returns curated picks

---

## Technical Architecture

### Tech Stack
- **Python 3.11+** — Best ecosystem for recommendation libraries
- **Typer** — Modern CLI framework (auto-generates help, clean syntax)
- **SQLite** — Local database via `sqlite3` (stdlib, no dependencies)
- **TMDb API** — Movie metadata (free tier)
- **Ollama + llama3.2** — Local LLM for mood queries (free, runs on M2)
- **Rich** — Pretty terminal output (colors, tables, spinners)

### Project Structure

```
flickpick/
├── flickpick/
│   ├── __init__.py
│   ├── cli.py            # Typer commands
│   ├── db.py             # SQLite operations
│   ├── tmdb.py           # TMDb API client
│   ├── recommender.py    # Recommendation logic
│   ├── llm.py            # Ollama integration
│   └── config.py         # API keys, settings
├── tests/
├── pyproject.toml        # Dependencies, packaging
└── README.md
```

### Database Schema

```sql
-- movies: cached TMDb data
CREATE TABLE movies (
    id INTEGER PRIMARY KEY,
    tmdb_id INTEGER UNIQUE NOT NULL,
    title TEXT NOT NULL,
    year INTEGER,
    genres TEXT,           -- JSON array
    plot TEXT,
    poster_url TEXT,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ratings: user's watch history
CREATE TABLE ratings (
    id INTEGER PRIMARY KEY,
    tmdb_id INTEGER NOT NULL,
    rating INTEGER,        -- 1-10, NULL if just watched
    watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tmdb_id) REFERENCES movies(tmdb_id)
);
```

### Recommendation Approach

1. **Content-based** (`like` command): TF-IDF on genres + plot keywords, cosine similarity
2. **Taste profile** (`suggest` command): Aggregate highly-rated genres/keywords, find unwatched matches
3. **LLM mood** (`mood` command): Pass query + top genres to Ollama, let it reason about matches

---

## Setup & First Run

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/flickpick.git
cd flickpick
pip install -e .
```

### First Run Wizard

```
$ flickpick

Welcome to flickpick! Let's get you set up.

1. TMDb API Key
   → Get one free at: https://www.themoviedb.org/settings/api
   → Paste your key: **********************
   ✓ Key saved to ~/.config/flickpick/config.toml

2. Ollama (for mood-based recommendations)
   → Installing Ollama... ✓
   → Pulling llama3.2 model... ✓ (2.0 GB)

3. Rate a few movies to build your taste profile
   → Search: inception
   → "Inception (2010)" - Your rating (1-10): 9
   → Add another? (y/n): y
   → Search: the matrix
   → "The Matrix (1999)" - Your rating (1-10): 10
   → Add another? (y/n): n

✓ Setup complete! You've rated 2 movies.

Try: flickpick suggest
```

### Config Locations
- Config: `~/.config/flickpick/config.toml`
- Database: `~/.local/share/flickpick/flickpick.db`

---

## Future Enhancements

After v1 ships successfully:

1. **TV series support** — seasons/episodes tracking
2. **Web UI** — FastAPI backend + simple frontend
3. **Watchlist** — `flickpick add "Dune 2"` for movies to watch
4. **"Where to watch"** — Integrate Watchmode API for streaming links
5. **Export** — Letterboxd/IMDb CSV export of ratings

---

## Dependencies

```toml
[project]
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "httpx>=0.25.0",        # Async HTTP for TMDb
    "scikit-learn>=1.3.0",  # TF-IDF, cosine similarity
    "ollama>=0.1.0",        # Ollama Python client
    "tomli>=2.0.0",         # Config parsing (Python <3.11)
    "tomli-w>=1.0.0",       # Config writing
]
```

---

## External Services

### TMDb API
- **URL:** https://api.themoviedb.org/3
- **Auth:** API key (free tier)
- **Rate limit:** ~40 requests/second
- **Signup:** https://www.themoviedb.org/settings/api

### Ollama
- **Model:** llama3.2 (3B parameters, ~2GB)
- **Install:** `brew install ollama` or `curl -fsSL https://ollama.com/install.sh | sh`
- **Pull model:** `ollama pull llama3.2`
