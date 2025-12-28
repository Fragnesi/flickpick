"""Test CLI commands."""

from typer.testing import CliRunner
from flickpick.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "flickpick" in result.stdout
    assert "0.1.0" in result.stdout
