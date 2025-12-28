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
