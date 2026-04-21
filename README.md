# MSAP-Sleep

Geometric reorganization of a pre-trained LLM via a sleep phase driven by thermodynamic pressure with life floors. See `docs/superpowers/specs/2026-04-21-msap-sleep-design.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest -m "not gpu"       # CPU-only tests
pytest                    # full suite (requires GPU)
```
