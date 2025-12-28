import httpx
from typing import Any

from flickpick.config import get_tmdb_api_key


BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def _parse_year(release_date: str | None) -> int | None:
    if not release_date:
        return None
    try:
        return int(release_date[:4])
    except (ValueError, IndexError):
        return None


class TMDbClient:

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_tmdb_api_key()
        if not self.api_key:
            raise ValueError(
                "TMDb API key not configured. Run 'flickpick setup' or set TMDB_API_KEY."
            )
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            params={"api_key": self.api_key},
            timeout=30.0,
        )
        self._genres: dict[int, str] | None = None

    async def _ensure_genres(self) -> dict[int, str]:
        if self._genres is None:
            response = await self._client.get("/genre/movie/list")
            response.raise_for_status()
            data = response.json()
            self._genres = {g["id"]: g["name"] for g in data["genres"]}
        return self._genres

    async def search(self, query: str) -> list[dict[str, Any]]:
        genres = await self._ensure_genres()
        response = await self._client.get(
            "/search/movie",
            params={"query": query},
        )
        response.raise_for_status()
        data = response.json()
        
        return [
            {
                "tmdb_id": movie["id"],
                "title": movie["title"],
                "year": _parse_year(movie.get("release_date")),
                "genres": [genres.get(gid, "Unknown") for gid in movie.get("genre_ids", [])],
                "plot": movie.get("overview"),
                "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
                "rating": movie.get("vote_average"),
            }
            for movie in data.get("results", [])
        ]

    async def get_movie(self, tmdb_id: int) -> dict[str, Any]:
        response = await self._client.get(f"/movie/{tmdb_id}")
        response.raise_for_status()
        movie = response.json()
        
        return {
            "tmdb_id": movie["id"],
            "title": movie["title"],
            "year": _parse_year(movie.get("release_date")),
            "genres": [g["name"] for g in movie.get("genres", [])],
            "plot": movie.get("overview"),
            "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
            "rating": movie.get("vote_average"),
            "runtime": movie.get("runtime"),
            "tagline": movie.get("tagline"),
        }

    async def get_popular(self, page: int = 1) -> list[dict[str, Any]]:
        genres = await self._ensure_genres()
        response = await self._client.get(
            "/movie/popular",
            params={"page": page},
        )
        response.raise_for_status()
        data = response.json()
        
        return [
            {
                "tmdb_id": movie["id"],
                "title": movie["title"],
                "year": _parse_year(movie.get("release_date")),
                "genres": [genres.get(gid, "Unknown") for gid in movie.get("genre_ids", [])],
                "plot": movie.get("overview"),
                "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
                "rating": movie.get("vote_average"),
            }
            for movie in data.get("results", [])
        ]

    async def discover(
        self,
        genres: list[int] | None = None,
        year: int | None = None,
        min_rating: float | None = None,
    ) -> list[dict[str, Any]]:
        genre_map = await self._ensure_genres()
        params: dict[str, Any] = {"sort_by": "vote_average.desc", "vote_count.gte": 100}
        if genres:
            params["with_genres"] = ",".join(str(g) for g in genres)
        if year:
            params["year"] = year
        if min_rating:
            params["vote_average.gte"] = min_rating
        
        response = await self._client.get("/discover/movie", params=params)
        response.raise_for_status()
        data = response.json()
        
        return [
            {
                "tmdb_id": movie["id"],
                "title": movie["title"],
                "year": _parse_year(movie.get("release_date")),
                "genres": [genre_map.get(gid, "Unknown") for gid in movie.get("genre_ids", [])],
                "plot": movie.get("overview"),
                "poster_url": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else None,
                "rating": movie.get("vote_average"),
            }
            for movie in data.get("results", [])
        ]

    async def close(self) -> None:
        await self._client.aclose()
