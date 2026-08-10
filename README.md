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

Paper: [`docs/paper/main.pdf`](docs/paper/main.pdf); source and the arXiv bundle in
[`docs/paper/`](docs/paper/).

> The artifact directories and the Hugging Face dataset repos are named `msap`, and some
> paths carry `RoMuLo`/`RoMuX` labels. Those are **historical names from an earlier line of
> work**, kept because the paper cites these paths and dataset names verbatim. They are not
> the subject of the paper.

## What is here

| | |
|---|---|
| [`docs/paper/`](docs/paper/) | LaTeX source, the built PDF, and the arXiv submission bundle |
| [`docs/protocols/`](docs/protocols/) | The seven pre-registration protocols, with an [index](docs/protocols/README.md) |
| `scripts/` | 77 scripts: the analyses below, plus the training, generation and probing code behind the paper's arms |
| `runs/` | 206 result artifacts — the outputs of those scripts, versioned |
| `outputs/` | 10 further analysis outputs: the generation-degeneration, truncation and loop-control audits behind the paper's generative chapter, and the evaluation lockfile |

## Setup

```bash
git clone https://github.com/seba-cascardo/higher-scores-not-a-better-model.git
cd higher-scores-not-a-better-model
pip install -r requirements.txt
```

Fourteen tracked tensors are stored as Git LFS objects, including the two
`mc_wknow_offset.pt` files that the off-axis decomposition reads. `.gitattributes` declares
them, so a clone made with git-lfs installed fetches them automatically. If you cloned
without it, install git-lfs and run `git lfs pull` — otherwise those paths are 133-byte
pointer files where the analysis expects a tensor, and the failure reads like a broken
command rather than an incomplete clone.

Fetching artifacts from the Hub additionally needs the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

`requirements.txt` covers every analysis in the table below; those scripts are CPU-only and
run anywhere. To re-run the *evaluations* rather than the analyses, install the exact stack
the runs were produced under — versions read off the environment block that lm-eval stamps
into every eval artifact:

```bash
pip install -r requirements-paper.txt
```

Evaluation ran on Python 3.12.13 / Ubuntu 24.04, one RTX PRO 6000 Blackwell (96 GB).

## Claim → command → artifact

Every headline number is re-derivable from a released artifact. The analysis scripts read
the evaluation JSONs and print the published quantities.

| Paper claim | Analysis script | Artifact it writes |
|---|---|---|
| Recovery fraction, paired bootstrap (ARC 0.6214, CI [0.510, 0.732]) | `l1_paired_bootstrap_recovery_fraction.py` | `runs/oq1_functional_axis/l1_paired_bootstrap.json`; `--metric acc` writes `_acc.json`, the raw-accuracy recomputation |
| Same-parameterization control (3 seeds, 36.2% ± 5.0) and objective × domain factorial (ratio 0.93 / 1.00) | `analyze_e4_factorial.py` | `runs/e4_factorial/e4_factorial_summary.json` |
| Qwen cross-family replication | `analyze_e6_qwen_spine.py` | `runs/e6_qwen_spine/e6_spine_summary.json` |
| Off-axis decomposition (parallel / perpendicular / random) | `analyze_offset_parperp_decomposition.py` | `runs/derisk/offset_parperp_decomposition.json` (Gemma), `runs/oq1_functional_axis/offset_parperp_qwen.json` (Qwen) |
| On-axis reference `W_know` (gold not linearly decodable options-absent) | `analyze_wknow_decodability.py` | `runs/gate2/wknow_decodability.json` |
| Per-item arbiter (base solves 124/124 under CoT; 442/444 on the full split) | `analyze_arbiter_arc_cot.py` | prints only — reads `runs/vinf_causal/arbiter_arc_cot_v2.json`, `arbiter_arc_sets{,_fullsplit}.json` |
| Margin rescaling mechanism (§5.9) | `analyze_threshold_density_mechanism.py` | `runs/e4_factorial/threshold_density_mechanism.json` |
| — its stratified permutation control | `audit_stratified_null_survivor.py` | `runs/e4_factorial/stratified_null_survivor_audit.json` |
| — its held-out validation (fit on half, score on the other half) | `analyze_affine_map_heldout.py` | `runs/e4_factorial/affine_map_heldout.json` |

The scripts' **outputs** are versioned in this repository; most of their **inputs** (the
per-item lm-eval JSONs and the direction tensors) are not, because of their size. Fetch them
per row, from public Hugging Face datasets under
[`sebacascardo87`](https://huggingface.co/sebacascardo87). Run every command from the
repository root.

### Recovery fraction (paired bootstrap)

The dataset stores these samples under `r5_align_p0/`, while the script globs
`runs/eval_bulletproof/`, so the download is followed by a relocation. The two arms differ
in one way that matters: the base arm's glob is recursive, so a nested snapshot directory
under `R1_base/` is fine, but the align arm's is not — its files must land directly under
`R1_align/runs__align_lora_control__r256/`. The script aborts unless each pattern resolves
to exactly one file.

```bash
hf download sebacascardo87/msap-r5-align-p0-20260628 --repo-type dataset \
  --include "r5_align_p0/R1_base/*" --include "r5_align_p0/R1_align/*" \
  --include "r5_align_p0/R1_v7mc.json" --include "r5_align_p0/R1_null.json" \
  --local-dir /tmp/msap_r5
mkdir -p runs/eval_bulletproof && cp -r /tmp/msap_r5/r5_align_p0/. runs/eval_bulletproof/
python scripts/l1_paired_bootstrap_recovery_fraction.py
python scripts/l1_paired_bootstrap_recovery_fraction.py --metric acc
```

### Same-parameterization control, factorial, margin rescaling, held-out validation

These share inputs. `analyze_e4_factorial.py` needs the first two datasets; the other three
additionally need the Qwen arms.

```bash
hf download sebacascardo87/msap-review-remediation-20260724 --repo-type dataset \
  --include "runs/e1_plain_ce/eval_*" --local-dir .
hf download sebacascardo87/msap-e4-e6-20260725 --repo-type dataset \
  --include "runs/e4_factorial/eval_ood_*" --include "runs/e6_qwen_spine/eval_*" --local-dir .
python scripts/analyze_e4_factorial.py
python scripts/analyze_threshold_density_mechanism.py
python scripts/audit_stratified_null_survivor.py
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

No download: all four inputs are versioned. This row runs from a clean checkout.

```bash
python scripts/analyze_arbiter_arc_cot.py
python scripts/analyze_arbiter_arc_cot.py --sets runs/vinf_causal/arbiter_arc_sets_fullsplit.json \
  --cot runs/vinf_causal/arbiter_arc_cot_fullsplit.json
```

### One way these can fail quietly

`analyze_threshold_density_mechanism.py` and `analyze_affine_map_heldout.py` read the
per-item `samples` block, not the aggregate `results` block, and they **skip** missing arms
instead of aborting. A partial download yields a thinner JSON with no error.

### One artifact here is not the paper's protocol

`runs/v7_lofit_gemma4_31b_chat/eval/eval_mc_full.json` is a **superseded in-house harness**,
kept as provenance for the released `offsets_mc.pt` and cited nowhere in the paper. Its
baseline differs from the canonical one on every task — gsm8k base `0.36` here against
`0.9775` under the paper's protocol — so its deltas are not comparable to the reported
ones, and its `+21.0` on gsm8k is not a counterexample to the reported `-25.0`. The
sibling [`README.txt`](runs/v7_lofit_gemma4_31b_chat/eval/README.txt) gives the full
comparison and the two ways to tell the harnesses apart from the file alone. Every number
in the paper comes from lm-eval result files.

## Models and provenance

Base models: [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) and
[`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).

Snapshot provenance is uneven and the paper says so rather than smoothing it over
(`docs/paper/sections/method.tex`): the harness resolved by `revision=main` and lm-eval
leaves `model_sha` empty, so **no result file carries an explicit pin**. Two Gemma
snapshots are recovered by hash from the cache paths the artifacts record; the headline
trio is dated rather than resolved, which is weaker evidence. Qwen is pinned by revision.
`runs/provenance/` holds the recovered map.

Model checkpoints and evaluation artifacts are released under
[`sebacascardo87`](https://huggingface.co/sebacascardo87).

## Pre-registration

The kill-rules and matched-control designs are dated protocol documents written *before*
the corresponding runs, collected under [`docs/protocols/`](docs/protocols/) with an index
that says what each covers and what its timestamp does and does not establish. They are
internal documents, not third-party registry entries, and the paper says so explicitly
where the term is used.

## Licence

Code in this repository is Apache-2.0 (see [`LICENSE`](LICENSE)). The third-party data is
not: released evaluation artifacts reproduce benchmark items verbatim, and each dataset's
terms — including ARC's share-alike condition — are itemised in
[`DATA-LICENSES.md`](DATA-LICENSES.md).

## Citation

```bibtex
@misc{cascardo2026higherscores,
  title  = {Higher Scores, Not A Better Model: A Large Multiple-Choice Lift That Is
            Mostly Inducible, Off-Axis, and Does Not Transfer to Generation},
  author = {Sebasti\'an Cascardo},
  year   = {2026}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).
