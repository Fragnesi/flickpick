"""CLI entry point."""

import typer

app = typer.Typer(
    name="flickpick",
    help="Movie recommendation system - track ratings and get personalized suggestions.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def callback():
    pass


@app.command()
def version():
    """Show version."""
    from flickpick import __version__
    from rich import print
    print(f"[bold]flickpick[/bold] v{__version__}")


if __name__ == "__main__":
    app()
