"""MORPH-DA command-line interface."""

from __future__ import annotations

import sys
try:
    import typer
except ImportError:
    print("Install typer: pip install typer", file=sys.stderr)
    sys.exit(1)

app = typer.Typer(help="MORPH-DA: Mutation-Grounded Metamorphic Verification")


@app.command()
def validate(
    seeds: list[int] = typer.Option([42, 7, 123], help="Data seeds for reference validation"),
    verbose: bool = typer.Option(True),
) -> None:
    """Validate all reference programs (Phase 1 exit condition)."""
    from morphda.tasks.factory import generate_task_set
    from morphda.tasks.validators import validate_task_set
    tasks = generate_task_set()
    _, summary = validate_task_set(tasks, seeds=seeds, verbose=verbose)
    status = "PASS" if summary["failed"] == 0 and summary["mr_violations"] == 0 else "FAIL"
    typer.echo(f"\nPhase 1 exit condition: {status}")
    raise typer.Exit(0 if status == "PASS" else 1)


@app.command()
def info() -> None:
    """Print benchmark summary."""
    from morphda.tasks.factory import generate_task_set, task_set_summary
    from morphda.relations import ALL_RELATIONS
    from morphda.mutations import ALL_OPERATORS
    tasks = generate_task_set()
    typer.echo(task_set_summary(tasks))
    typer.echo(f"Relations: {len(ALL_RELATIONS)}")
    typer.echo(f"Mutation operators: {len(ALL_OPERATORS)}")


@app.command()
def evaluate(
    program: str = typer.Argument(help="Path to Python file with analyze(tables) function"),
    task_spec: str = typer.Argument(help="Path to task spec YAML"),
    data_dir: str = typer.Argument(help="Path to data directory with scenario CSVs"),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate one candidate program against a task spec."""
    typer.echo(f"Evaluating {program} on {task_spec} (seed={seed})...")
    typer.echo("(Full evaluation pipeline — see run_verification.py for batch mode)")


if __name__ == "__main__":
    app()
