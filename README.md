# RoMuLo (Routed Multi-LoFiT)

Research codebase for routed-modular adapters that specialize an LLM per task class without destroying its base capability. Repo and Python package are still named `msap` for historical reasons (rename: 2026-05-04).

**Status**: see [`CLAUDE.md`](CLAUDE.md) for project context and the latest `docs/superpowers/specs/*-NEXT-SESSION-START-HERE.md` for the active session handoff.

**Targets**: Gemma 4 31B IT (primary), Qwen 14B IT (cross-family).

**Previous era**: V1–V6 ("MSAP-Sleep") explored geometric reorganization of a pre-trained LLM via a thermodynamic-pressure sleep phase with life floors. Pivoted to LoFiT routing in V7+ after V1–V6 failed to generalize. Original design spec: [`docs/superpowers/specs/2026-04-21-msap-sleep-design.md`](docs/superpowers/specs/2026-04-21-msap-sleep-design.md).

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
