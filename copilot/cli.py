"""
cli.py — Typer CLI for the Incident Intelligence Copilot.

Commands:
  init    Load topology, seed incidents, create vector index, embed training set.
  ask     Diagnose a new incident symptom.
  ablate  Run the ablation study (vector vs graph vs hybrid) on held-out incidents.
"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="GraphRAG Incident-Intelligence Copilot")
console = Console()


@app.command()
def init(
    n: int = typer.Option(25, "--incidents", "-n", help="Number of synthetic incidents to seed"),
    holdout: float = typer.Option(0.2, "--holdout", help="Fraction held out for evaluation"),
) -> None:
    """Bootstrap: load topology, seed incidents, build vector index, embed training set."""
    from copilot.graph import (
        create_vector_index,
        embed_incidents,
        load_topology,
        seed_incidents,
    )

    typer.echo("Loading topology ...")
    load_topology()

    typer.echo("Seeding incidents ...")
    seed_incidents(n=n, holdout_frac=holdout)

    typer.echo("Creating vector index ...")
    create_vector_index()

    typer.echo("Embedding training incidents (this may take ~30s on first run) ...")
    embed_incidents()

    typer.echo("\nDone. System ready.")
    typer.echo(f"  Try: python -m copilot.cli ask --symptom 'checkout throwing 503s' --service checkout")


@app.command()
def ask(
    symptom: str = typer.Option(..., "--symptom", "-s", help="Incident symptom description"),
    service: str = typer.Option(..., "--service", help="Name of the affected service"),
) -> None:
    """Diagnose a new incident: retrieve evidence and synthesize a grounded explanation."""
    from copilot.synthesize import diagnose

    typer.echo("\n--- Diagnosis ---\n")
    typer.echo(diagnose(symptom, service))
    typer.echo("")


@app.command()
def ablate(
    n: int = typer.Option(3, "--n", help="Top-N candidates for hit-rate metric"),
    chaos: bool = typer.Option(False, "--chaos", help="Use chaos-labelled incidents instead of holdout synthetic set"),
) -> None:
    """
    Ablation study: compare vector-only, graph-only, and hybrid retrieval.
    Use --chaos to evaluate on real chaos-injected incidents.
    """
    from copilot.eval import ablation, build_chaos_testset, build_testset

    if chaos:
        typer.echo("Loading chaos-labelled test set ...")
        testset = build_chaos_testset()
    else:
        typer.echo("Loading holdout synthetic test set ...")
        testset = build_testset()

    if not testset:
        typer.echo(
            "[error] No holdout incidents found. Run 'init' first.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Running ablation on {len(testset)} held-out incidents ...\n")
    results = ablation(testset, n=n)

    table = Table(title=f"Ablation Results  ({len(testset)} held-out incidents)")
    table.add_column("Retrieval Mode", style="cyan", min_width=16)
    table.add_column(f"Hit-Rate@{n}", style="green", justify="right")
    table.add_column("MRR", style="yellow", justify="right")

    for mode, metrics in results.items():
        table.add_row(mode, str(metrics[f"hit_rate@{n}"]), str(metrics["mrr"]))

    console.print(table)


if __name__ == "__main__":
    app()
