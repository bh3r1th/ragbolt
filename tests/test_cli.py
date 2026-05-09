import json
from pathlib import Path

from typer.testing import CliRunner

from ragbolt.cli.main import app

runner = CliRunner()


def test_cli_run_accepted(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    trace_path = tmp_path / "trace.json"
    corpus_path.write_text(
        json.dumps([
            {"chunk_id": "c1", "text": "BM25 ranking retrieval function applies", "source": "a.txt"},
            {"chunk_id": "c2", "text": "alpha beta gamma unrelated tokens", "source": "b.txt"},
            {"chunk_id": "c3", "text": "another distinct topic words here", "source": "c.txt"},
        ]),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["run", str(corpus_path), "ranking function", "--output", str(trace_path)],
    )
    assert result.exit_code == 0
    assert "ACCEPTED" in result.output
    assert trace_path.exists()


def test_cli_run_missing_corpus(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "nope.json"), "query"])
    assert result.exit_code == 1


def test_cli_eval_missing_report(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
