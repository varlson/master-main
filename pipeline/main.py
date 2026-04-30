from __future__ import annotations

import sys

from pipeline.cli import parse_args
from pipeline.workflows.orchestrator import run_command


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_command(args)


if __name__ == "__main__":
    main()
