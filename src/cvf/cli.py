"""Command line interface: `cvf --help`."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Annotated

import typer

from cvf.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Customer purchase propensity and 90-day revenue forecasting.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def _parse_date(value: str | None) -> date | None:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _setup_logging(verbose)


@app.command()
def ingest(force_download: bool = False) -> None:
    """Download the source extract and load it into raw.transactions."""
    from cvf.ingest import load_raw

    n = load_raw(get_settings(), force_download=force_download)
    typer.echo(f"Loaded {n:,} raw rows")


@app.command()
def build() -> None:
    """Rebuild staging tables, the feature store and labels."""
    from cvf.pipeline import build as build_

    summary = build_(get_settings())
    typer.echo(f"Built {int(summary['customers'].sum()):,} customer-snapshot rows "
               f"over {len(summary)} snapshots")


@app.command()
def tune(n_trials: int = 0) -> None:
    """Search LightGBM hyper-parameters on purged inner folds."""
    from cvf.pipeline import load_dataset
    from cvf.tuning import run_tuning

    settings = get_settings()
    result = run_tuning(load_dataset(settings), settings, n_trials=n_trials or None)
    typer.echo(f"Tuned params kept for: {', '.join(result) or 'none (defaults won)'}")


@app.command()
def train() -> None:
    """Backtest, fit production models, register them and write the model report."""
    from cvf.pipeline import train as train_

    bundle = train_(get_settings())
    typer.echo(f"Registered model {bundle.version}")
    for task, m in bundle.metadata["holdout_metrics"].items():
        typer.echo(f"  {task}: " + ", ".join(f"{k}={v}" for k, v in m.items()))


@app.command()
def experiments() -> None:
    """Re-run the modelling ablations and write reports/experiments.md."""
    from cvf.experiments import run_experiments
    from cvf.pipeline import load_dataset

    settings = get_settings()
    typer.echo(run_experiments(load_dataset(settings), settings))


@app.command()
def score(as_of: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None) -> None:
    """Score the active customer base and write mart.customer_scores."""
    from cvf.pipeline import score as score_

    scores, drift = score_(get_settings(), _parse_date(as_of))
    typer.echo(f"Scored {len(scores):,} customers as of {scores['snapshot_date'].iloc[0]}; "
               f"{int(scores['contact'].sum()):,} selected for contact; "
               f"expected revenue £{scores['expected_revenue_90d'].sum():,.0f}")
    flagged = drift[drift["status"] != "ok"]
    if not flagged.empty:
        typer.echo("Drift:\n" + flagged.to_string(index=False))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the scoring API."""
    import uvicorn

    uvicorn.run("cvf.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
