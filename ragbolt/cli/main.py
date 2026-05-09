import typer
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


def _load_config(config: Path) -> dict:
    config_dict = _default_config()
    if config.exists():
        try:
            loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(e) from e
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a YAML mapping")
        config_dict.update(loaded)
    return config_dict


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
    trace: Path = typer.Argument(..., help="Path to rag_trace.json"),
    report: Path = typer.Option(
        Path("eval_report.json"),
        help="Path to write eval_report.json",
    ),
):
    """Generate and print eval report from a trace file."""
    try:
        from ragbolt.eval.report import load_and_build_report, summary_lines

        r = load_and_build_report(trace, report)
        for line in summary_lines(r):
            typer.echo(line)
        typer.echo(f"\nReport written to: {report}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("explain")
def explain(
    trace: Path = typer.Argument(..., help="Path to rag_trace.json"),
    run_id: str = typer.Option(None, help="Explain a specific run_id only"),
):
    """Print human-readable explanation of a trace file."""
    try:
        from ragbolt.trace.explain import load_trace, explain_event, explain_trace

        events = load_trace(trace)
        if run_id:
            matched = [e for e in events if e["run_id"] == run_id]
            if not matched:
                typer.echo(f"Error: run_id {run_id} not found in trace", err=True)
                raise typer.Exit(1)
            events = matched
        for line in explain_trace(events):
            typer.echo(line)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("ingest")
def ingest(
    source: Path = typer.Argument(..., help="File or directory to ingest"),
    output: Path = typer.Option(Path("corpus.json"), help="Output corpus JSON path"),
    chunk_size: int = typer.Option(512, help="Max words per chunk"),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recurse into subdirectories",
    ),
):
    """Ingest text files into a ragbolt corpus JSON."""
    try:
        from ragbolt.ingest import ingest_file, ingest_directory, write_corpus
        from ragbolt.core.policy import Corpus

        if source.is_dir():
            corpus = ingest_directory(source, chunk_size, recursive)
        elif source.is_file():
            chunks = ingest_file(source, chunk_size)
            corpus = Corpus(chunks=chunks, corpus_id=source.stem)
        else:
            typer.echo(f"Error: {source} not found", err=True)
            raise typer.Exit(1)
        write_corpus(corpus, output)
        typer.echo(f"Ingested {len(corpus.chunks)} chunks → {output}")
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("batch")
def batch(
    corpus: Path = typer.Argument(..., help="Path to corpus JSON"),
    queries: Path = typer.Argument(..., help="Path to queries .jsonl or .txt"),
    config: Path = typer.Option(Path("config.yaml"), help="YAML config path"),
    output: Path = typer.Option(Path("rag_trace.json"), help="Trace output path"),
    provider: str = typer.Option("stub", help="stub, anthropic, openai"),
    retriever: str = typer.Option("bm25", help="bm25 or hybrid"),
    verifier: str = typer.Option("stub", help="stub or production"),
):
    """Run ragbolt on multiple queries from a file."""
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

        from ragbolt.core.batch import load_queries, run_batch

        query_list = load_queries(queries)
        orchestrator = RepairOrchestrator(
            corpus_obj,
            config_dict,
            generation_provider,
            ega,
            retriever=retriever_obj,
        )
        emitter = TraceEmitter(output)
        result = run_batch(query_list, orchestrator, emitter, corpus_obj.corpus_id)
        emitter.flush()
        typer.echo(f"Batch complete: {result.total} queries")
        for outcome, count in result.outcome_counts.items():
            typer.echo(f"  {outcome:<20}: {count}")
        typer.echo(f"Trace written to: {output}")
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("serve")
def serve(
    corpus: Path = typer.Argument(..., help="Path to corpus JSON"),
    config: Path = typer.Option(Path("config.yaml"), help="YAML config path"),
    provider: str = typer.Option("stub", help="stub, anthropic, openai"),
    retriever: str = typer.Option("bm25", help="bm25 or hybrid"),
    verifier: str = typer.Option("stub", help="stub or production"),
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
):
    """Start ragbolt as a REST API server."""
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "Error: ragbolt serve requires uvicorn. Install with: pip install fastapi uvicorn[standard]",
            err=True,
        )
        raise typer.Exit(1)
    try:
        from ragbolt.serve import create_app

        config_dict = _load_config(config)
        application = create_app(corpus, config_dict, provider, retriever, verifier)
        typer.echo(f"ragbolt serving on http://{host}:{port}")
        typer.echo("  POST /query  — run pipeline")
        typer.echo("  GET  /health — corpus status")
        typer.echo("  GET  /trace  — view trace file")
        uvicorn.run(application, host=host, port=port)
    except (ValueError, FileNotFoundError, ImportError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("calibrate")
def calibrate(
    trace: Path = typer.Argument(..., help="Path to rag_trace.json"),
    config: Path = typer.Option(Path("config.yaml"), help="Current config path"),
    target_abstain_rate: float = typer.Option(0.05, help="Target abstain rate"),
    target_fail_rate: float = typer.Option(0.10, help="Target fail rate"),
    apply: bool = typer.Option(False, "--apply", help="Write changes to config.yaml"),
):
    """Recommend threshold adjustments based on trace history."""
    try:
        from ragbolt.core.calibrate import calibrate_from_trace, format_calibration

        config_dict = _load_config(config)
        result = calibrate_from_trace(trace, target_abstain_rate, target_fail_rate)
        for line in format_calibration(result):
            typer.echo(line)
        if apply:
            import yaml

            config_dict["bm25_min_score"] = result.recommended_bm25_min_score
            config_dict["unsupported_ratio_threshold"] = (
                result.recommended_unsupported_ratio_threshold
            )
            config.write_text(yaml.dump(config_dict), encoding="utf-8")
            typer.echo(f"\nConfig updated: {config}")
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("export")
def export(
    trace: Path = typer.Argument(..., help="Path to rag_trace.json"),
    endpoint: str = typer.Option(
        "http://localhost:4318/v1/traces",
        help="OTLP HTTP endpoint",
    ),
    service_name: str = typer.Option("ragbolt", help="OTEL service name"),
):
    """Export trace to OpenTelemetry collector."""
    from ragbolt.trace.otel import load_and_export

    success, errors = load_and_export(trace, endpoint, service_name)
    if errors:
        for e in errors:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Exported {success} span(s) to {endpoint}")


if __name__ == "__main__":
    app()
