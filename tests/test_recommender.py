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
