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


def test_cli_unknown_provider(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [{"chunk_id": "c1", "text": "retrieval ranking bm25", "source": "a.txt"}]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["run", str(corpus_path), "query", "--provider", "bad"],
    )
    assert result.exit_code == 1


def test_cli_unknown_retriever(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [{"chunk_id": "c1", "text": "retrieval ranking bm25", "source": "a.txt"}]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["run", str(corpus_path), "query", "--retriever", "bad"],
    )
    assert result.exit_code == 1


def _run_to_trace(tmp_path: Path) -> Path:
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
    return trace_path


def test_cli_explain_accepted(tmp_path: Path) -> None:
    trace_path = _run_to_trace(tmp_path)
    result = runner.invoke(app, ["explain", str(trace_path)])
    assert result.exit_code == 0
    assert "ACCEPTED" in result.output
    assert "Run ID" in result.output


def test_cli_explain_missing_trace(tmp_path: Path) -> None:
    result = runner.invoke(app, ["explain", str(tmp_path / "nope.json")])
    assert result.exit_code == 1


def test_cli_explain_run_id_filter(tmp_path: Path) -> None:
    trace_path = _run_to_trace(tmp_path)
    events = json.loads(trace_path.read_text(encoding="utf-8"))
    run_id = events[0]["run_id"]
    result = runner.invoke(app, ["explain", str(trace_path), "--run-id", run_id])
    assert result.exit_code == 0
    assert run_id in result.output


def test_cli_eval_generates_report(tmp_path: Path) -> None:
    trace_path = _run_to_trace(tmp_path)
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["eval", str(trace_path), "--report", str(report_path)],
    )
    assert result.exit_code == 0
    assert report_path.exists()
    assert "ACCEPTED" in result.output
    assert "Total cases" in result.output
