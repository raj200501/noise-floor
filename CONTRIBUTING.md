# Contributing

## Setup

```bash
make setup     # uv sync --locked (Python 3.11+)
make doctor    # what this machine can run
make ci        # lint, typecheck, test, offline demo
```

## Rules

- Types and docstrings on every public function; tests first. `mypy --strict` and `ruff` must be
  clean, and every `type: ignore` carries a reason after it (`# type: ignore[code]  # why`).
- No hand-typed result numbers in documents: every reported number comes from a committed script
  run on a banked artifact under `artifacts/`, and `make claimcheck` enforces this.
- Confirmatory analyses are preregistered in `prereg/` before any data exists. Frozen
  preregistrations are never edited; changes are new dated amendment files.
- Synthetic or mock data is labeled as such everywhere.
- Plain writing: precise, scoped claims, no hype words, no em dashes.
- Record substitutions and interpretations in `DECISIONS.md` before acting on them, and every
  changed or withdrawn claim in `CORRECTIONS.md`.
