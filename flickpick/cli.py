"""CLI entry point for flickpick."""

import asyncio
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

from flickpick import __version__
from flickpick.config import get_tmdb_api_key, save_config
from flickpick.db import Database
from flickpick.tmdb import TMDbClient
from flickpick.recommender import build_taste_profile, find_similar_movies, score_by_profile
from flickpick.llm import get_mood_recommendations, check_ollama_available

app = typer.Typer(
    name="flickpick",
    help="Movie recommendation system - track ratings and get personalized suggestions.",
    no_args_is_help=True,
)
console = Console()


def run_async(coro):
    """Run async function in sync context."""
    return asyncio.run(coro)


@app.command()
def version():
    """Show version."""
    print(f"[bold]flickpick[/bold] v{__version__}")


@app.command()
def setup():
    """Interactive setup wizard."""
    console.print(Panel.fit("[bold]Welcome to flickpick![/bold]\nLet's get you set up.", border_style="blue"))
    
    console.print("\n[bold]1. TMDb API Key[/bold]")
    console.print("   Get one free at: [link]https://www.themoviedb.org/settings/api[/link]")
    
    api_key = Prompt.ask("   Paste your API key")
    if api_key:
        save_config({"tmdb_api_key": api_key.strip()})
        console.print("   [green]OK[/green] API key saved!")
    
    console.print("\n[bold]2. Ollama (for mood-based recommendations)[/bold]")
    available, message = check_ollama_available()
    if available:
        console.print(f"   [green]OK[/green] {message}")
    else:
        console.print(f"   [yellow]![/yellow] {message}")
        console.print("   Install: [code]brew install ollama && ollama pull llama3.2[/code]")
    
    console.print("\n[green]OK Setup complete![/green]")
    console.print("Try: [code]flickpick search inception[/code]")


@app.command()
def search(query: str = typer.Argument(..., help="Movie title to search for")):
    """Search for movies on TMDb."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def do_search():
        client = TMDbClient(api_key)
        try:
            results = await client.search(query)
            return results
        finally:
            await client.close()
    
    results = run_async(do_search())
    
    if not results:
        console.print(f"No movies found for '{query}'")
        return
    
    table = Table(title=f"Search results for '{query}'")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("Rating", width=6)
    
    for i, movie in enumerate(results[:10], 1):
        genres = ", ".join(movie.get("genres", [])[:3])
        rating = f"{movie.get('rating', 0):.1f}" if movie.get("rating") else "-"
        table.add_row(
            str(i),
            movie["title"],
            str(movie.get("year", "-")),
            genres,
            rating,
        )
    
    console.print(table)


@app.command()
def info(title: str = typer.Argument(..., help="Movie title")):
    """Show detailed movie information."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def get_info():
        client = TMDbClient(api_key)
        try:
            results = await client.search(title)
            if not results:
                return None
            movie = await client.get_movie(results[0]["tmdb_id"])
            return movie
        finally:
            await client.close()
    
    movie = run_async(get_info())
    
    if not movie:
        console.print(f"Movie not found: '{title}'")
        return
    
    console.print(Panel.fit(
        f"[bold]{movie['title']}[/bold] ({movie.get('year', 'N/A')})\n\n"
        f"[dim]{movie.get('tagline', '')}[/dim]\n\n"
        f"[bold]Genres:[/bold] {', '.join(movie.get('genres', []))}\n"
        f"[bold]Runtime:[/bold] {movie.get('runtime', 'N/A')} min\n"
        f"[bold]Rating:[/bold] {movie.get('rating', 'N/A')}/10\n\n"
        f"[bold]Plot:[/bold]\n{movie.get('plot', 'No plot available.')}",
        border_style="blue",
    ))


@app.command()
def rate(
    title: str = typer.Argument(..., help="Movie title"),
    rating: int = typer.Argument(..., help="Rating from 1-10"),
):
    """Rate a movie (1-10)."""
    if not 1 <= rating <= 10:
        console.print("[red]Error:[/red] Rating must be between 1 and 10")
        raise typer.Exit(1)
    
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def find_and_rate():
        client = TMDbClient(api_key)
        try:
            results = await client.search(title)
            if not results:
                return None
            movie = await client.get_movie(results[0]["tmdb_id"])
            return movie
        finally:
            await client.close()
    
    movie = run_async(find_and_rate())
    
    if not movie:
        console.print(f"Movie not found: '{title}'")
        raise typer.Exit(1)
    
    db = Database()
    db.cache_movie(
        tmdb_id=movie["tmdb_id"],
        title=movie["title"],
        year=movie.get("year"),
        genres=movie.get("genres"),
        plot=movie.get("plot"),
        poster_url=movie.get("poster_url"),
    )
    db.add_rating(movie["tmdb_id"], rating)
    
    console.print(f"[green]OK[/green] Rated [bold]{movie['title']}[/bold] ({movie.get('year', 'N/A')}) - {rating}/10")


@app.command()
def watched(title: str = typer.Argument(..., help="Movie title")):
    """Mark a movie as watched (without rating)."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def find_movie():
        client = TMDbClient(api_key)
        try:
            results = await client.search(title)
            if not results:
                return None
            movie = await client.get_movie(results[0]["tmdb_id"])
            return movie
        finally:
            await client.close()
    
    movie = run_async(find_movie())
    
    if not movie:
        console.print(f"Movie not found: '{title}'")
        raise typer.Exit(1)
    
    db = Database()
    db.cache_movie(
        tmdb_id=movie["tmdb_id"],
        title=movie["title"],
        year=movie.get("year"),
        genres=movie.get("genres"),
        plot=movie.get("plot"),
        poster_url=movie.get("poster_url"),
    )
    db.add_rating(movie["tmdb_id"], rating=None)
    
    console.print(f"[green]OK[/green] Marked [bold]{movie['title']}[/bold] ({movie.get('year', 'N/A')}) as watched")


@app.command()
def history():
    """Show your watch history."""
    db = Database()
    ratings = db.get_all_ratings()
    
    if not ratings:
        console.print("No movies in your history yet. Try: [code]flickpick rate 'Inception' 9[/code]")
        return
    
    table = Table(title="Your Watch History")
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Rating", width=8)
    table.add_column("Genres")
    
    for entry in ratings:
        rating_str = f"{entry['rating']}/10" if entry.get("rating") else "[dim]watched[/dim]"
        genres = ", ".join(entry.get("genres", [])[:3]) if entry.get("genres") else "-"
        table.add_row(
            entry["title"],
            str(entry.get("year", "-")),
            rating_str,
            genres,
        )
    
    console.print(table)


@app.command(name="like")
def like_movies(
    titles: list[str] = typer.Argument(..., help="Movie title(s) to find similar movies for"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recommendations"),
):
    """Find movies similar to the ones you specify."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    async def get_similar():
        client = TMDbClient(api_key)
        try:
            source_movies = []
            for title in titles:
                results = await client.search(title)
                if results:
                    movie = await client.get_movie(results[0]["tmdb_id"])
                    source_movies.append(movie)
            
            if not source_movies:
                return None, []
            
            candidates = await client.get_popular()
            
            return source_movies, candidates
        finally:
            await client.close()
    
    source_movies, candidates = run_async(get_similar())
    
    if not source_movies:
        console.print("Could not find any of the specified movies")
        raise typer.Exit(1)
    
    source_ids = {m["tmdb_id"] for m in source_movies}
    candidates = [c for c in candidates if c["tmdb_id"] not in source_ids]
    
    db = Database()
    watched_ids = {r["tmdb_id"] for r in db.get_all_ratings()}
    candidates = [c for c in candidates if c["tmdb_id"] not in watched_ids]
    
    similar = find_similar_movies(source_movies, candidates, limit=limit)
    
    source_titles = ", ".join(m["title"] for m in source_movies)
    console.print(f"\n[bold]Movies similar to:[/bold] {source_titles}\n")
    
    table = Table()
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("Match", width=6)
    
    for i, movie in enumerate(similar, 1):
        genres = ", ".join(movie.get("genres", [])[:3])
        score = movie.get("similarity_score", 0)
        match_pct = f"{score * 100:.0f}%"
        table.add_row(
            str(i),
            movie["title"],
            str(movie.get("year", "-")),
            genres,
            match_pct,
        )
    
    console.print(table)


@app.command()
def suggest(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recommendations"),
):
    """Get personalized recommendations based on your ratings."""
    api_key = get_tmdb_api_key()
    if not api_key:
        console.print("[red]Error:[/red] TMDb API key not configured. Run [code]flickpick setup[/code]")
        raise typer.Exit(1)
    
    db = Database()
    rated_movies = db.get_rated_movies()
    
    if len(rated_movies) < 3:
        console.print("Rate at least 3 movies first for personalized recommendations.")
        console.print("Try: [code]flickpick rate 'Movie Name' 8[/code]")
        return
    
    profile = build_taste_profile(rated_movies)
    
    async def get_candidates():
        client = TMDbClient(api_key)
        try:
            return await client.get_popular()
        finally:
            await client.close()
    
    candidates = run_async(get_candidates())
    
    watched_ids = {r["tmdb_id"] for r in db.get_all_ratings()}
    candidates = [c for c in candidates if c["tmdb_id"] not in watched_ids]
    
    recommendations = score_by_profile(candidates, profile)[:limit]
    
    console.print(f"\n[bold]Your top genres:[/bold] {', '.join(profile['top_genres'])}\n")
    
    table = Table(title="Recommended for You")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("TMDb", width=6)
    
    for i, movie in enumerate(recommendations, 1):
        genres = ", ".join(movie.get("genres", [])[:3])
        rating = f"{movie.get('rating', 0):.1f}" if movie.get("rating") else "-"
        table.add_row(
            str(i),
            movie["title"],
            str(movie.get("year", "-")),
            genres,
            rating,
        )
    
    console.print(table)


@app.command()
def mood(
    query: str = typer.Argument(..., help="Describe what you're in the mood for"),
):
    """Get recommendations based on your mood (uses AI)."""
    available, message = check_ollama_available()
    if not available:
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(1)
    
    db = Database()
    rated_movies = db.get_rated_movies()
    profile = build_taste_profile(rated_movies)
    
    with console.status("[bold blue]Thinking...[/bold blue]"):
        response = run_async(get_mood_recommendations(query, profile))
    
    console.print(Panel.fit(
        f"[bold]Mood:[/bold] {query}\n\n{response}",
        title="AI Recommendations",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
