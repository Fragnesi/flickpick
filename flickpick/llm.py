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
        
        # Skip lines that look like prose/preamble
        if line.endswith(":") or line.lower().startswith(("based on", "here are", "i recommend")):
            continue
        
        # Remove numbering (1., 2., etc.)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Remove bullet points
        line = re.sub(r"^[-•*]\s*", "", line)
        # Remove quotes
        line = line.strip('"\'')
        
        if not line:
            continue
        
        # Extract title - everything before " - " description if present
        if " - " in line:
            title = line.split(" - ", 1)[0].strip()
        else:
            title = line.strip()
        
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
