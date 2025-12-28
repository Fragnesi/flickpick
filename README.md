# flickpick

A CLI movie recommendation system that tracks your ratings and gives personalized suggestions.

## Features

- **Track movies** - Rate what you've watched, build your taste profile
- **"Movies like X"** - Find similar movies based on genres and plot
- **AI-powered mood search** - "something tense but not too violent" -> personalized picks
- **Local-first** - All data stored locally, no account needed

## Installation

```bash
git clone https://github.com/Fragnesi/flickpick.git
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
