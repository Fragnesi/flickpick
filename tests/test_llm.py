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
