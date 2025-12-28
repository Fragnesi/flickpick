import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import httpx


@pytest.fixture
def mock_tmdb_response():
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
    return {
        "genres": [
            {"id": 18, "name": "Drama"},
            {"id": 53, "name": "Thriller"},
            {"id": 28, "name": "Action"},
        ]
    }


@pytest.fixture
def mock_movie_detail_response():
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
    from flickpick.tmdb import TMDbClient
    
    with patch("flickpick.tmdb.get_tmdb_api_key", return_value=None):
        with pytest.raises(ValueError, match="TMDb API key not configured"):
            TMDbClient()


@pytest.mark.asyncio
async def test_search_movies(mock_tmdb_response, mock_genre_response):
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
            mock_get.side_effect = [mock_genre_resp, mock_response]
            
            results = await client.search("Fight Club")
            
            assert len(results) == 1
            assert results[0]["title"] == "Fight Club"
            assert results[0]["year"] == 1999
            assert results[0]["tmdb_id"] == 550


@pytest.mark.asyncio
async def test_get_movie_details(mock_movie_detail_response):
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
    from flickpick.tmdb import _parse_year
    
    assert _parse_year("1999-10-15") == 1999
    assert _parse_year("2023-01-01") == 2023
    assert _parse_year("") is None
    assert _parse_year(None) is None
