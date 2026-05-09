import typer
import json
from collections import Counter
from pathlib import Path

import yaml

from ragbolt.core.generator import StubGenerationProvider
from ragbolt.core.orchestrator import RepairOrchestrator
from ragbolt.core.policy import load_corpus
from ragbolt.trace.emitter import TraceEmitter
from ragbolt.verify.stub import StubEGAVerifier

app = typer.Typer(name="ragbolt", help="Failure-aware RAG repair layer.")


def _default_config() -> dict:
    return {
        "bm25_min_score": 0.30,
        "overlap_min_jaccard": 0.15,
        "unsupported_ratio_threshold": 0.25,
        "top_k": 5,
        "top_k_max": 10,
        "context_reduction_mode": "chunk",
    }


@app.command("run")
def run(
    corpus: Path = typer.Argument(..., help="Path to corpus JSON file"),
    query: str = typer.Argument(..., help="Query string"),
    config: Path = typer.Option(Path("config.yaml"), help="YAML config path"),
    output: Path = typer.Option(Path("rag_trace.json"), help="Trace output path"),
    provider: str = typer.Option("stub", help="Generation provider: stub, anthropic, or openai"),
    retriever: str = typer.Option("bm25", help="bm25 or hybrid"),
    verifier: str = typer.Option("stub", help="stub or production"),
):
    """Run ragbolt on a single query against a corpus."""
    config_dict = _default_config()
    if config.exists():
        try:
            loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            typer.echo("Error: Config file must contain a YAML mapping", err=True)
            raise typer.Exit(1)
        config_dict.update(loaded)
    try:
        corpus_obj = load_corpus(corpus)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    try:
        if provider == "stub":
            generation_provider = StubGenerationProvider()
        elif provider == "anthropic":
            from ragbolt.core.providers import AnthropicGenerationProvider

            generation_provider = AnthropicGenerationProvider(config_dict)
        elif provider == "openai":
            from ragbolt.core.providers import OpenAIGenerationProvider

            generation_provider = OpenAIGenerationProvider(config_dict)
        else:
            typer.echo("Error: unknown provider", err=True)
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    try:
        retriever_obj = None
        if retriever == "hybrid":
            from ragbolt.core.hybrid import HybridRunner

            retriever_obj = HybridRunner(corpus_obj, config_dict)
        elif retriever == "bm25":
            retriever_obj = None
        else:
            typer.echo("Error: unknown retriever", err=True)
            raise typer.Exit(1)

        if verifier == "production":
            from ragbolt.verify.production import ProductionEGAVerifier

            ega = ProductionEGAVerifier(config_dict)
        elif verifier == "stub":
            ega = StubEGAVerifier(config_dict)
        else:
            typer.echo("Error: unknown verifier", err=True)
            raise typer.Exit(1)

        orchestrator = RepairOrchestrator(
            corpus_obj,
            config_dict,
            generation_provider,
            ega,
            retriever=retriever_obj,
        )
        result = orchestrator.run(query)
        emitter = TraceEmitter(output)
        run_id = emitter.record(
            corpus_id=corpus_obj.corpus_id,
            query=query,
            failure_classes=result.failure_classes,
            repair_attempts=result.repair_attempts,
            outcome=result.outcome,
            top_score=result.top_score,
            chunks_retrieved=result.chunks_retrieved,
            unsupported_ratio=result.unsupported_ratio,
            chunk_ids=result.chunk_ids,
            raw_top_score=result.raw_top_score,
        )
        emitter.flush()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Outcome: {result.outcome.value}  run_id: {run_id}  trace: {output}")


@app.command("eval")
def eval(
    report: Path = typer.Argument(..., help="Path to eval_report.json"),
):
    """Print eval report summary to stdout."""
    if not report.exists():
        typer.echo(f"Error: {report} not found", err=True)
        raise typer.Exit(1)
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    try:
        if isinstance(payload, dict) and "per_case" in payload and isinstance(payload["per_case"], list):
            per_case = payload["per_case"]
        elif isinstance(payload, list):
            per_case = payload
        else:
            typer.echo("Error: Invalid report JSON format", err=True)
            raise typer.Exit(1)

        typer.echo(f"per_case: {len(per_case)}")
        outcomes = Counter()
        for item in per_case:
            if isinstance(item, dict):
                outcome = item.get("outcome")
                if outcome is not None:
                    outcomes[str(outcome)] += 1
        for key, count in outcomes.items():
            typer.echo(f"{key}: {count}")
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
