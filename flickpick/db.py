import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class Database:

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            from flickpick.config import get_db_path
            db_path = get_db_path()
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY,
                tmdb_id INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                year INTEGER,
                genres TEXT,
                plot TEXT,
                poster_url TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY,
                tmdb_id INTEGER NOT NULL,
                rating INTEGER,
                watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tmdb_id) REFERENCES movies(tmdb_id)
            );

            CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies(tmdb_id);
            CREATE INDEX IF NOT EXISTS idx_ratings_tmdb_id ON ratings(tmdb_id);
        """)
        self.conn.commit()

    def cache_movie(
        self,
        tmdb_id: int,
        title: str,
        year: int | None = None,
        genres: list[str] | None = None,
        plot: str | None = None,
        poster_url: str | None = None,
    ) -> None:
        genres_json = json.dumps(genres) if genres else None
        self.conn.execute(
            """
            INSERT INTO movies (tmdb_id, title, year, genres, plot, poster_url)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id) DO UPDATE SET
                title = excluded.title,
                year = excluded.year,
                genres = excluded.genres,
                plot = excluded.plot,
                poster_url = excluded.poster_url,
                cached_at = CURRENT_TIMESTAMP
            """,
            (tmdb_id, title, year, genres_json, plot, poster_url),
        )
        self.conn.commit()

    def get_movie(self, tmdb_id: int) -> dict[str, Any] | None:
        cursor = self.conn.execute(
            "SELECT * FROM movies WHERE tmdb_id = ?", (tmdb_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_movie(row)

    def _row_to_movie(self, row: sqlite3.Row) -> dict[str, Any]:
        movie = dict(row)
        if movie.get("genres"):
            movie["genres"] = json.loads(movie["genres"])
        return movie

    def add_rating(self, tmdb_id: int, rating: int | None = None) -> None:
        self.conn.execute("DELETE FROM ratings WHERE tmdb_id = ?", (tmdb_id,))
        self.conn.execute(
            "INSERT INTO ratings (tmdb_id, rating) VALUES (?, ?)",
            (tmdb_id, rating),
        )
        self.conn.commit()

    def get_all_ratings(self) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT r.*, m.title, m.year, m.genres, m.plot, m.poster_url
            FROM ratings r
            JOIN movies m ON r.tmdb_id = m.tmdb_id
            ORDER BY r.watched_at DESC
            """
        )
        return [self._row_to_rating(row) for row in cursor.fetchall()]

    def _row_to_rating(self, row: sqlite3.Row) -> dict[str, Any]:
        rating = dict(row)
        if rating.get("genres"):
            rating["genres"] = json.loads(rating["genres"])
        return rating

    def is_watched(self, tmdb_id: int) -> bool:
        cursor = self.conn.execute(
            "SELECT 1 FROM ratings WHERE tmdb_id = ?", (tmdb_id,)
        )
        return cursor.fetchone() is not None

    def get_rated_movies(self) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT r.*, m.title, m.year, m.genres, m.plot, m.poster_url
            FROM ratings r
            JOIN movies m ON r.tmdb_id = m.tmdb_id
            WHERE r.rating IS NOT NULL
            ORDER BY r.rating DESC
            """
        )
        return [self._row_to_rating(row) for row in cursor.fetchall()]

    def get_all_cached_movies(self) -> list[dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM movies")
        return [self._row_to_movie(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()
