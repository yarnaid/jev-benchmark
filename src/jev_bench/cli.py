"""Command-line entry point: `serve`, `generate`, `run`.

Heavy modules (uvicorn, FastAPI, numpy, httpx2) are imported inside the commands so `--help`
stays fast.

Constants:
    LOOPBACK
Functions:
    serve: start the web UI and API.
    generate: create one generation and wait for it.
    run: run one benchmark column on generations and wait for it.
"""

from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="Jev benchmark: generate emails, run columns, serve the UI.",
)

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Interface to bind; keep loopback unless you know why.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes (development).")] = False,
) -> None:
    """Serve the web UI and the JSON API."""
    import uvicorn
    from rich.console import Console

    if host not in LOOPBACK:
        Console(stderr=True).print(
            f"[yellow]Warning:[/] binding to {host}: "
            "API keys entered in the UI travel over plain HTTP."
        )
    uvicorn.run(
        "jev_bench.web.app:create_default_app", factory=True, host=host, port=port, reload=reload
    )


@app.command()
def generate(
    count: Annotated[int, typer.Option(min=1, max=2000, help="Number of emails.")] = 200,
    name: Annotated[str, typer.Option(help="Generation name.")] = "generation",
    seed: Annotated[int | None, typer.Option(min=0, help="Seed; random when omitted.")] = None,
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="Generator model (repeatable); defaults to the config mix."),
    ] = None,
) -> None:
    """Generate a new set of synthetic emails with reference answers."""
    import asyncio

    from jev_bench.log_setup import configure_logging

    configure_logging()

    from jev_bench import cli_jobs
    from jev_bench.generation.launcher import GenerationRequest

    request = GenerationRequest(
        name=name, count=count, seed=seed, models=tuple(model) if model else None
    )
    raise typer.Exit(asyncio.run(cli_jobs.generate_and_wait(request)))


@app.command()
def run(
    column: Annotated[str, typer.Argument(help="Column id from config/benchmark.toml.")],
    generations: Annotated[
        list[str], typer.Option("--generation", "-g", help="Generation id (repeatable).")
    ],
    model: Annotated[
        str | None, typer.Option(help="Model id; defaults to the column default.")
    ] = None,
    mode: Annotated[
        str | None, typer.Option(help="per_email or all_in_one (chat columns only).")
    ] = None,
) -> None:
    """Run one benchmark column on the given generations."""
    import asyncio

    from jev_bench.log_setup import configure_logging

    configure_logging()

    from jev_bench import cli_jobs

    raise typer.Exit(asyncio.run(cli_jobs.run_and_wait(column, generations, model, mode)))
