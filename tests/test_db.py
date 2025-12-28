import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def test_db(tmp_path):
    from flickpick.db import Database
    
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


def test_database_creates_tables(test_db):
    cursor = test_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert "movies" in tables
    assert "ratings" in tables


def test_cache_movie(test_db):
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
    assert test_db.get_movie(99999) is None


def test_add_rating(test_db):
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999)
    
    test_db.add_rating(tmdb_id=550, rating=9)
    
    ratings = test_db.get_all_ratings()
    assert len(ratings) == 1
    assert ratings[0]["tmdb_id"] == 550
    assert ratings[0]["rating"] == 9


def test_add_watched_without_rating(test_db):
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999)
    test_db.add_rating(tmdb_id=550, rating=None)
    
    ratings = test_db.get_all_ratings()
    assert len(ratings) == 1
    assert ratings[0]["rating"] is None


def test_get_all_ratings_with_movie_info(test_db):
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999, genres=["Drama"])
    test_db.add_rating(tmdb_id=550, rating=9)
    
    ratings = test_db.get_all_ratings()
    assert ratings[0]["title"] == "Fight Club"
    assert ratings[0]["year"] == 1999


def test_is_watched(test_db):
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999)
    
    assert not test_db.is_watched(550)
    test_db.add_rating(tmdb_id=550, rating=8)
    assert test_db.is_watched(550)


def test_get_rated_movies_for_recommendations(test_db):
    test_db.cache_movie(tmdb_id=550, title="Fight Club", year=1999, genres=["Drama"])
    test_db.cache_movie(tmdb_id=680, title="Pulp Fiction", year=1994, genres=["Crime"])
    
    test_db.add_rating(tmdb_id=550, rating=9)
    test_db.add_rating(tmdb_id=680, rating=None)
    
    rated = test_db.get_rated_movies()
    assert len(rated) == 1
    assert rated[0]["tmdb_id"] == 550
