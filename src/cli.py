from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from .logging_config import configure_logging

logger = logging.getLogger(__name__)


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from .project_pipeline import run_pipeline

    summary = run_pipeline()
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_notebooks(args: argparse.Namespace) -> int:
    from .notebook_builder import build_eda_notebook, build_modeling_notebook, execute_notebook
    from .config import ROOT_DIR

    notebooks_dir = ROOT_DIR / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building EDA notebook...")
    build_eda_notebook()
    logger.info("Building modeling notebook...")
    build_modeling_notebook()

    if args.execute:
        for name in ("01_eda.ipynb", "02_modeling.ipynb"):
            logger.info("Executing %s", name)
            execute_notebook(notebooks_dir / name)

    return 0


def _add_verbose(p: argparse.ArgumentParser) -> None:
    """Register the shared --verbose flag on every parser/subparser."""
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v=INFO, -vv=DEBUG).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retail-sales-pipeline",
        description="Retail sales prediction project CLI.",
    )
    _add_verbose(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pipeline = sub.add_parser("pipeline", help="Run the full ML pipeline.")
    _add_verbose(p_pipeline)
    p_pipeline.set_defaults(func=_cmd_pipeline)

    p_nb = sub.add_parser("notebooks", help="Generate the project notebooks.")
    _add_verbose(p_nb)
    p_nb.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebooks after generation (slow).",
    )
    p_nb.set_defaults(func=_cmd_notebooks)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    configure_logging(level=level)

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
