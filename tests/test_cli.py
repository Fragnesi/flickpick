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
