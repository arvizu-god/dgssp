# `dgssp.cli`

## What the module does

A thin command-line front end over `dgssp.experiments` and `dgssp.runtime`,
installed as the `dgssp-run` entry point. Three subcommands: run one instance,
sweep random instances, or save IBM credentials.

The CLI deliberately exposes no ZNE knobs beyond on/off — mitigation studies
belong in a notebook or script where the configuration is visible and version
controlled.

## Contents

| Name | Kind |
|---|---|
| `_parse_items` | internal function |
| `build_parser` | function |
| `main` | function |

---

### `_parse_items(text) -> list[int]` — internal

- **Input:** `text: str`, a comma-separated list such as `"3,5,-2,10"`. Spaces are tolerated.
- **Output:** the parsed integers.
- **Raises:** `argparse.ArgumentTypeError` if any entry is not an integer.

---

### `build_parser() -> argparse.ArgumentParser`

- **Input:** none.
- **Output:** the parser, with `--version` and the three subcommands attached.

**`solve`** — run one instance on the ideal simulator.
`--items` (required), `--target` (required), `--shots`, `--seed`, `--top`.

**`batch`** — sweep random instances.
`--n-instances`, `--n-items`, `--max-value`, `--shots`, `--seed`, `--json`.

**`account`** — save credentials.
`--token` (required), `--instance` (CRN), `--name`.

---

### `main(argv=None) -> int`

Runs the CLI.

- **Input:** `argv: Sequence[str] | None`, defaulting to `sys.argv[1:]`.
- **Output:** the process exit status — `0` on success, `1` on a handled error.

`solve` prints the instance, the qubit and iteration counts, the number of
exact solutions, the measured `P(solution)`, and the top outcomes with their
decoded subsets — exact solutions marked with `*`. `batch` prints the JSON
summary and optionally writes the full `BatchResult`. `account` persists
credentials; this is the one place in the library that writes them, and only
when you ask.
