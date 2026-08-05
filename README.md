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
git clone --branch feature/v8 https://github.com/seba-cascardo/msap.git && cd msap
git lfs pull
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**The branch matters.** The release lives on `feature/v8`; the repository's default branch
(`master`) predates it and carries neither the analysis scripts, the versioned result
artifacts, nor the licence files. A default `git clone` cannot run any row of the table below.

On Windows, clone into a short path (e.g. `C:\src`) or enable
`git config --global core.longpaths true`: the deepest tracked evaluation paths are 200
characters, so a long parent directory pushes them past the 260-character limit and checkout
fails file by file.

**`git lfs pull` is not optional.** Fourteen tracked artifacts are stored as Git LFS objects,
including the two `mc_wknow_offset.pt` files that the off-axis decomposition row reads. There
is no `.gitattributes` in the tree, so nothing in a fresh checkout announces this: without
git-lfs those paths are ~130-byte text pointers where the analysis scripts expect a tensor,
and the failure reads like a broken command rather than an incomplete clone. (The recovery-
fraction row below touches no LFS object — its inputs are JSON fetched from the Hub.)

Fetching artifacts additionally needs the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
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

| Paper claim | Analysis script | Artifact it writes |
|---|---|---|
| Recovery fraction, paired bootstrap (ARC 0.6214, CI [0.510, 0.732]) | `l1_paired_bootstrap_recovery_fraction.py` | `runs/oq1_functional_axis/l1_paired_bootstrap.json` (and `_acc.json` for the raw-accuracy recomputation) |
| Same-parameterization control (3 seeds, 36.2% ± 5.0) and objective × domain factorial (ratio 0.93 / 1.00) | `analyze_e4_factorial.py` | `runs/e4_factorial/e4_factorial_summary.json` |
| Qwen cross-family replication | `analyze_e6_qwen_spine.py` | `runs/e6_qwen_spine/e6_spine_summary.json` |
| Off-axis decomposition (parallel / perpendicular / random) | `analyze_offset_parperp_decomposition.py` | `runs/derisk/offset_parperp_decomposition.json` (Gemma), `runs/oq1_functional_axis/offset_parperp_qwen.json` (Qwen) |
| On-axis reference `W_know` (gold not linearly decodable options-absent) | `analyze_wknow_decodability.py` | `runs/gate2/wknow_decodability.json` |
| Per-item arbiter (base solves 124/124 under CoT; 442/444 on the full split) | `analyze_arbiter_arc_cot.py` | prints only — reads `runs/vinf_causal/arbiter_arc_cot_v2.json`, `arbiter_arc_sets{,_fullsplit}.json` |
| Margin rescaling mechanism (§5.9) | `analyze_threshold_density_mechanism.py` | `runs/e4_factorial/threshold_density_mechanism.json`, `stratified_null_survivor_audit.json` |
| — its held-out validation (fit on half, score on the other half) | `analyze_affine_map_heldout.py` | `runs/e4_factorial/affine_map_heldout.json` |

The scripts' **outputs** are versioned in this repository; most of their **inputs** (the
per-item lm-eval JSONs and the direction tensors) are not, because of their size. Fetch them
per row, from public Hugging Face datasets under
[`sebacascardo87`](https://huggingface.co/sebacascardo87). Run every command from the
repository root.

### Recovery fraction (paired bootstrap)

The dataset stores these samples under `r5_align_p0/`, while the script globs
`runs/eval_bulletproof/`, so the download is followed by a relocation. The nested
snapshot directory must be preserved — the glob matches through it, and the script aborts
unless each pattern resolves to exactly one file.

```bash
hf download sebacascardo87/msap-r5-align-p0-20260628 --repo-type dataset \
  --include "r5_align_p0/R1_base/*" --include "r5_align_p0/R1_align/*" \
  --include "r5_align_p0/R1_v7mc.json" --include "r5_align_p0/R1_null.json" \
  --local-dir /tmp/msap_r5
mkdir -p runs/eval_bulletproof && cp -r /tmp/msap_r5/r5_align_p0/. runs/eval_bulletproof/
python scripts/l1_paired_bootstrap_recovery_fraction.py
```

### Same-parameterization control, factorial, margin rescaling, held-out validation

These four share inputs. `analyze_e4_factorial.py` needs the first two datasets;
`analyze_threshold_density_mechanism.py` and `analyze_affine_map_heldout.py` additionally
need the Qwen arms.

```bash
hf download sebacascardo87/msap-review-remediation-20260724 --repo-type dataset \
  --include "runs/e1_plain_ce/eval_*" --local-dir .
hf download sebacascardo87/msap-e4-e6-20260725 --repo-type dataset \
  --include "runs/e4_factorial/eval_ood_*" --include "runs/e6_qwen_spine/eval_*" --local-dir .
python scripts/analyze_e4_factorial.py
python scripts/analyze_threshold_density_mechanism.py
python scripts/analyze_affine_map_heldout.py
```

### Qwen cross-family replication

```bash
hf download sebacascardo87/msap-e4-e6-20260725 --repo-type dataset \
  --include "runs/e6_qwen_spine/*" --local-dir .
python scripts/analyze_e6_qwen_spine.py
```

### Off-axis decomposition

Configured entirely through environment variables, and the only row that reads Git LFS
objects (see *Setup*).

```bash
hf download sebacascardo87/msap-oq1-vg-20260622 --repo-type dataset \
  --include "functional_directions.pt" --local-dir runs
MSAP_ROOT=. python scripts/analyze_offset_parperp_decomposition.py

# Qwen arm
hf download sebacascardo87/msap-qwen-offaxis-inputs-20260706 --repo-type dataset \
  --include "functional_directions_qwen.pt" --local-dir runs/oq1_functional_axis
MSAP_ROOT=. FUNC_DIRS=runs/oq1_functional_axis/functional_directions_qwen.pt \
  MC_OFFSETS=runs/v7_lofit_qwen14b/offsets_mc.pt \
  WKNOW_OFFSET=runs/vinf_causal_qwen/mc_wknow_offset.pt \
  PARPERP_OUT=runs/oq1_functional_axis/offset_parperp_qwen.json \
  python scripts/analyze_offset_parperp_decomposition.py
```

### On-axis reference `W_know`

Both flags are required; the script has no path defaults.

```bash
hf download sebacascardo87/msap-gate2-faithful-20260624 --repo-type dataset \
  --include "gate2/promptfinal_residual_arc.pt" --local-dir runs
python scripts/analyze_wknow_decodability.py \
  --resid runs/gate2/promptfinal_residual_arc.pt \
  --out runs/gate2/wknow_decodability.json
```

### Per-item arbiter

No download: all four inputs are versioned.

```bash
python scripts/analyze_arbiter_arc_cot.py
python scripts/analyze_arbiter_arc_cot.py --sets runs/vinf_causal/arbiter_arc_sets_fullsplit.json \
  --cot runs/vinf_causal/arbiter_arc_cot_fullsplit.json
```

### Two ways these can fail quietly

- `analyze_threshold_density_mechanism.py` and `analyze_affine_map_heldout.py` read the
  per-item `samples` block, not the aggregate `results` block, and they **skip** missing arms
  instead of aborting. A partial download yields a thinner JSON with no error.
- `hf download` takes repeated `--exclude`/`--include` flags. Passing several values to one
  flag is not an error: the extra values are parsed as positional *filenames to download*, so
  the command silently fetches roughly the opposite of what was intended.

Model checkpoints and evaluation artifacts are released on the Hugging Face Hub under
[`sebacascardo87`](https://huggingface.co/sebacascardo87). Base models: `google/gemma-4-31b-it`
and `Qwen/Qwen2.5-14B-Instruct` — see [`docs/paper/`](docs/paper/) for the pinned revisions
used in every run. Third-party dataset licences are itemised in
[`DATA-LICENSES.md`](DATA-LICENSES.md).

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
