"""Command line interface to run examples."""
import argparse
import runpy
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run TACDMPC examples")
    parser.add_argument("name", help="Example module name without package prefix")
    args = parser.parse_args()
    runpy.run_module(f"examples.{args.name}", run_name="__main__")


if __name__ == "__main__":  # pragma: no cover
    cli()
