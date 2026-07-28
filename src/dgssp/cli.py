"""
cli.py

A thin command-line front end over :mod:`dgssp.experiments` and
:mod:`dgssp.api`.

Installed as the ``dgssp-run`` entry point.  Three subcommands:

* ``solve``   -- run one instance given on the command line;
* ``batch``   -- sweep random instances and write a JSON report;
* ``account`` -- save IBM Quantum credentials (the one place in the library
  that writes them, and only when you ask).

The CLI deliberately exposes no ZNE knobs beyond on/off: mitigation studies
belong in a notebook or script where the configuration is visible and version
controlled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__


def _parse_items(text: str) -> list[int]:
    """
    Parse a comma-separated item list, tolerating spaces and negatives.

    Parameters
    ----------
    text:
        e.g. ``"3,5,-2,10"``.

    Returns
    -------
    list[int]
        The parsed items.

    Raises
    ------
    argparse.ArgumentTypeError
        If any entry is not an integer.
    """
    try:
        return [int(tok) for tok in text.replace(" ", "").split(",") if tok]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse items {text!r}; expected a comma-separated list "
            "of integers."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser, with the ``solve``, ``batch`` and ``account`` subcommands
        attached.
    """
    parser = argparse.ArgumentParser(
        prog="dgssp-run",
        description="Draper-Grover Subset Sum solver.",
    )
    parser.add_argument("--version", action="version", version=f"dgssp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="Run one instance on the ideal simulator.")
    solve.add_argument("--items", type=_parse_items, required=True,
                       help="Comma-separated integers, e.g. 3,5,-2,10")
    solve.add_argument("--target", type=int, required=True, help="Target sum.")
    solve.add_argument("--shots", type=int, default=10_000)
    solve.add_argument("--seed", type=int, default=None, help="Simulator seed.")
    solve.add_argument("--top", type=int, default=5,
                       help="How many outcomes to print.")

    batch = sub.add_parser("batch", help="Run a sweep of random instances.")
    batch.add_argument("--n-instances", type=int, default=5)
    batch.add_argument("--n-items", type=int, default=3)
    batch.add_argument("--max-value", type=int, default=20)
    batch.add_argument("--shots", type=int, default=10_000)
    batch.add_argument("--seed", type=int, default=0)
    batch.add_argument("--json", dest="json_path", default=None,
                       help="Write the full BatchResult to this path.")

    account = sub.add_parser("account", help="Save IBM Quantum credentials.")
    account.add_argument("--token", required=True)
    account.add_argument("--instance", default=None, help="Instance CRN.")
    account.add_argument("--name", default=None, help="Saved-account name.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the command-line interface.

    Parameters
    ----------
    argv:
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: ``0`` on success, ``1`` on a handled error.
    """
    args = build_parser().parse_args(argv)

    if args.command == "solve":
        from .decoding import bitstring_to_solution
        from .experiments import BatchConfig, run_batch
        from .instance import SubsetSumInstance

        instance = SubsetSumInstance(items=args.items, target=args.target, name="cli")
        result = run_batch(
            [instance],
            BatchConfig(shots=args.shots, seed_simulator=args.seed),
        ).results[0]

        print(f"instance          : {instance.items} -> {instance.target}")
        print(f"qubits / iterations: {result.n_qubits} / {result.iterations}")
        print(f"exact solutions   : {result.num_solutions}")
        print(f"P(solution)       : {result.solution_probability:.4f}")
        print("top outcomes:")
        ranked = sorted(result.distribution.items(), key=lambda kv: -kv[1])
        for bit, prob in ranked[: args.top]:
            sol = bitstring_to_solution(bit, instance)
            mark = "*" if sol.is_exact else " "
            print(f"  {mark} {bit}  p={prob:.4f}  subset={sol.subset} sum={sol.total}")
        return 0

    if args.command == "batch":
        from .experiments import BatchConfig, run_random_batch

        batch = run_random_batch(
            args.n_instances,
            args.n_items,
            max_value=args.max_value,
            seed=args.seed,
            config=BatchConfig(shots=args.shots, seed_simulator=args.seed),
        )
        print(json.dumps(batch.summary(), indent=2))
        if args.json_path:
            path = batch.to_json(args.json_path)
            print(f"wrote {path}")
        return 0

    if args.command == "account":
        from .runtime import save_account

        save_account(args.token, args.instance, name=args.name)
        print("Credentials saved.")
        return 0

    return 1  # pragma: no cover - argparse enforces a valid subcommand


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
