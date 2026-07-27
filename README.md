# Higher Scores, Not A Better Model

Code, evaluation protocols and result artifacts for the paper *"Higher Scores, Not A
Better Model: A Large Multiple-Choice Lift That Is Mostly Inducible, Off-Axis, and Does
Not Transfer to Generation"*.

**What the paper is about.** Contrastive-correctness parameter-efficient fine-tuning (the
LoFiT/ITI family) buys a large cold multiple-choice lift on a frontier 31B instruction
model — ARC-Challenge +35.0, HellaSwag +17.0, Winogrande +19.0, TruthfulQA-mc1 +37.25.
We ask what that lift *is* and answer with a quantified causal attribution: much of it is
inducible without the contrastive objective, the recovery is off the measured
one-dimensional correctness direction, and it does not transfer to generation. A
benchmark gain of this kind should not be read as a capability gain.

Paper source: [`docs/paper/`](docs/paper/) (LaTeX; `main.pdf` is the built document).

> The repo and the Python package are named `msap`, and much of the tree carries
> `RoMuLo`/`RoMuX` labels. Those are **historical names kept to avoid churn**, not the
> subject of the paper — see [`CLAUDE.md`](CLAUDE.md) for the project's history.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

To reproduce the paper's numbers, install the exact stack the runs were produced under
(versions read off the environment block that lm-eval stamps into every eval artifact):

```bash
pip install -e ".[paper]"
```

That pins `torch==2.12.0` (cu130), `transformers==5.8.0`, `lm-eval==0.4.12` and
`huggingface_hub<1.14`. Evaluation ran on Python 3.12.13 / Ubuntu 24.04, one RTX PRO 6000
Blackwell (96 GB); the analysis scripts below are CPU-only and run anywhere.

## Claim → command → artifact

Every headline number is re-derivable from a released artifact. The analysis scripts read
the evaluation JSONs and print the published quantities.

| Paper claim | Command | Artifact |
|---|---|---|
| Recovery fraction, paired bootstrap (ARC 0.6214, CI [0.510, 0.732]) | `python scripts/l1_paired_bootstrap_recovery_fraction.py` | `runs/oq1_functional_axis/l1_paired_bootstrap.json` (and `_acc.json` for the raw-accuracy recomputation) |
| Same-parameterization control, 3 seeds (36.2% ± 5.0) | `python scripts/analyze_e4_factorial.py` | `runs/e1_plain_ce/eval_lifts_seed{0,1,2}.json` |
| Objective × domain factorial (ratio 0.93 / 1.00) | `python scripts/analyze_e4_factorial.py` | `runs/e4_factorial/e4_factorial_summary.json` |
| Qwen cross-family replication | `python scripts/analyze_e6_qwen_spine.py` | `runs/e6_qwen_spine/e6_spine_summary.json` |
| Off-axis decomposition (parallel / perpendicular / random) | `python scripts/analyze_offset_parperp_decomposition.py` | `runs/derisk/offset_parperp_decomposition.json` (Gemma), `runs/oq1_functional_axis/offset_parperp_qwen.json` (Qwen) |
| On-axis reference `W_know` | `python scripts/analyze_wknow_decodability.py` | `runs/e6_qwen_spine/eval_mc_wknow_offset.json` |
| Per-item arbiter (base solves 124/124 under CoT; 442/444 on the full split) | `python scripts/analyze_arbiter_arc_cot.py` | `runs/vinf_causal/arbiter_arc_cot.json`, `arbiter_arc_cot_fullsplit.json` |
| Margin rescaling mechanism (§5.9) | `python scripts/analyze_threshold_density_mechanism.py` | `runs/e4_factorial/threshold_density_mechanism.json` |
| — its held-out validation (fit on half, score on the other half) | `python scripts/analyze_affine_map_heldout.py` | `runs/e4_factorial/affine_map_heldout.json` |
| Stratified-null audit of the surviving cell | (written by the mechanism script) | `runs/e4_factorial/stratified_null_survivor_audit.json` |

Model checkpoints and evaluation artifacts are released on the Hugging Face Hub under
[`sebacascardo87`](https://huggingface.co/sebacascardo87). Base models: `google/gemma-4-31b-it`
and `Qwen/Qwen2.5-14B-Instruct` — see [`docs/paper/`](docs/paper/) for the pinned revisions
used in every run.

## Pre-registration

The headline kill-rules and matched-control designs are dated protocol documents
versioned in this repository *before* the corresponding runs (git-timestamped), under
[`docs/superpowers/specs/`](docs/superpowers/specs/). They are internal and verifiable
from the release history, not third-party registry entries; the paper states this
explicitly where the term is used.

## Tests

```bash
pytest -m "not gpu"       # CPU-only tests
pytest                    # full suite (requires GPU)
```
