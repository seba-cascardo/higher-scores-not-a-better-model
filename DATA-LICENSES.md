# Third-party dataset licenses

The code in this repository is Apache-2.0 (see `LICENSE`). **This file covers the
third-party data**, which is not: several of these datasets are non-commercial, and
several declare no license at all.

Scope: benchmark items, corpora and derived artifacts that ship under `runs/` and
`outputs/`. Read the row for a dataset before reusing anything derived from it.

---

## Attribution required and NOT yet met

These four are the reason this file exists. The obligation is attribution (and, for the
ShareAlike ones, share-alike on derivatives); the repository shipped derived artifacts
without carrying the notice. **The rows below are that attribution.**

| Dataset | License | What it requires | Met? |
|---|---|---|---|
| **ARC** (AI2 Reasoning Challenge) | CC-BY-SA-4.0 | Attribution + ShareAlike on derivatives | **Now, by this file.** Previously not carried |
| **WikiText-103** | CC-BY-SA-3.0 | Attribution + ShareAlike on derivatives | **Now, by this file.** Previously not carried |
| **FineWeb** | ODC-BY-1.0 | Attribution; the ODC-BY notice must travel | **Now, by this file.** Previously not carried |
| **WildChat-1M** | ODC-BY-1.0 | Attribution; the ODC-BY notice must travel | **Now, by this file.** Previously not carried — see the caveat below |

- ARC — Clark et al., *Think you have Solved Question Answering?*, AI2, 2018.
- WikiText-103 — Merity et al., *Pointer Sentinel Mixture Models*, Salesforce Research, 2016.
- FineWeb — Penedo et al., HuggingFaceFW, 2024.
- WildChat-1M — Zhao et al., AllenAI, 2024.

WikiText is present as *content* inside artifacts whose filenames do not say so — e.g.
`runs/oq1_functional_axis/freq_control_r1e3_full_c4rerun.json`, which despite the `c4` in
its name is a WikiText-103 run. A filename scan does not find these rows; this table was
built from content.

## Redistributable with attribution, NON-COMMERCIAL

Fine to keep in a public repo with credit, but **derivatives may not be used
commercially**. This is a real restriction on downstream users, not a formality.

| Dataset | License | What it requires | Met? |
|---|---|---|---|
| **SciQ** (7 files) | CC-BY-NC-3.0 | Attribution; **non-commercial only** | Yes, by this file |
| **AIME 2025 / 2026** | CC-BY-NC-SA-4.0 | Attribution; **non-commercial**; ShareAlike | Yes, by this file |

- SciQ — Welbl et al., *Crowdsourcing Multiple Choice Science Questions*, AI2, 2017.
- AIME — Mathematical Association of America competition problems, as redistributed by
  the respective HuggingFace dataset cards.

## Permissive — redistributable with attribution

These carry the benchmark items that appear verbatim inside released evaluation artifacts
(`samples_*.jsonl` and the `samples` block of lm-eval result JSONs).

| Dataset | License | What it requires | Met? |
|---|---|---|---|
| **HellaSwag** | MIT | Attribution; carry the notice | Yes, by this file |
| **WinoGrande** | CC-BY (dataset; the codebase is Apache-2.0) | Attribution | Yes, by this file |
| **TruthfulQA** | Apache-2.0 | Attribution; carry the notice | Yes, by this file |
| **SocialIQA** | CC-BY-4.0 | Attribution | Yes, by this file |
| **PIQA** | AFL-3.0 (Academic Free License v3.0) | Attribution; carry the notice | Yes, by this file |
| **MMLU** | MIT | Attribution | Yes, by this file |
| **MMLU-Pro** | MIT | Attribution | Yes, by this file |
| **GSM8K** | MIT | Attribution | Yes, by this file |
| **`hails/bigbench`** (mirror of `google/BIG-bench`) | Apache-2.0 (upstream) | Attribution | Yes, by this file |

- HellaSwag — Zellers et al., 2019. `github.com/rowanz/hellaswag` (`LICENSE`: MIT, © Rowan Zellers).
- WinoGrande — Sakaguchi et al., AI2, 2019. `github.com/allenai/winogrande`: *"The dataset is
  licensed under CC-BY"*, codebase Apache-2.0.
- TruthfulQA — Lin et al., 2021. HF card `truthfulqa/truthful_qa`, `license: apache-2.0`.
- SocialIQA — Sap et al., AI2, 2019. HF card `allenai/social_i_qa`, Licensing Information: CC-BY-4.0.
- PIQA — Bisk et al., 2019. `yonatanbisk.com/piqa`: Academic Free License v3.0.
- MMLU — Hendrycks et al., 2021 (`cais/mmlu`). MMLU-Pro — Wang et al., 2024 (`TIGER-Lab/MMLU-Pro`).
- GSM8K — Cobbe et al., 2021 (`openai/gsm8k`).
- BIG-bench — Srivastava et al., 2022. `github.com/google/BIG-bench` (`LICENSE`: Apache-2.0).

> **Why these were previously recorded as UNKNOWN.** The earlier audit read the **license tag**
> in each card's YAML frontmatter. Several of these declare the license only in the card
> **body**, under *Licensing Information* — `allenai/social_i_qa` and `Rowan/hellaswag` both do —
> and WinoGrande and PIQA declare it upstream rather than on the Hub at all. A tag sweep reports
> those as unlicensed. Resolved 2026-08-05 by reading the card bodies and the upstream
> `LICENSE` files, two independent sources per dataset where available.

## UNDETERMINED — still not resolved

**Declared undetermined, not assumed permissive.** Absence of a stated license is not a grant.

| Dataset | License | What it requires | Met? |
|---|---|---|---|
| **PopQA** | **UNDETERMINED** — card silent, no upstream dataset licence found | Undetermined | **Cannot be determined** |
| **TriviaQA** | **UNDETERMINED** — none located | Undetermined | **Cannot be determined** |

PopQA's questions are templated from Wikidata tuples (Wikidata is CC0) and the associated code
repository (`AlexTMallen/adaptive-retrieval`) is MIT, but neither statement is a licence grant
for the dataset itself. Both datasets belong to abandoned experimental lines; no number in the
paper depends on them.

## GATED — not redistributed, purged from history

These three prohibit redistribution. No file carrying their items is present in this
repository, in the working tree or in any commit of its history — the release was cut with
`filter-repo` from a larger private repository, and the cut is verifiable by content:
scanning every blob that has ever existed here returns no occurrence of the review-stage
column names, the EWoK template fields, or the benchmark canary strings. Reproducing results
that use these datasets requires obtaining them from the original source under its own terms.

| Dataset | License | What it requires | Met? |
|---|---|---|---|
| **GPQA-Diamond** | Gated; redistribution prohibited | Obtain from source; do not redistribute items | Yes — 27 files purged, not shipped |
| **EWoK** | Gated; training data may not be redistributed | Obtain from source | Yes — 2 files purged, not shipped |
| **LIMA** | Gated | Obtain from source | Yes — 2 files purged, not shipped |

---

## Not verified (declared, not assumed)

- **If WildChat-1M turns out to be gated** rather than ODC-BY, the 11 files the audit
  identified under `runs/mixed_class_corpus/` would move from "attribution" to **blocking**.
  This was not verified. Those files are not part of this release — the directory is not in
  this repository — so the question does not bear on anything shipped here.
- Large binaries hosted on HuggingFace (activation `.pt`, LoRA `.safetensors`) were not
  opened byte by byte; their license verdict rests on the logs that produced them.
- The audit covered the 20 HuggingFace repos relevant to the paper, not all 49 in the
  account.

## Provenance

Built from a licence audit of the project's artifacts that classified them **by content**,
not by filename. The distinction mattered, and is the reason the table below is trustworthy:
files named `kr_*` turned out to hold ARC items, a file named `*_c4rerun` held WikiText, and
the file whose name most suggested GPQA carried no review-stage columns at all. A filename
sweep would have got all three wrong in both directions.

Updated 2026-08-05 during the Hugging Face release pass: the seven datasets previously listed
as UNKNOWN were re-checked against card bodies and upstream `LICENSE` files, and five resolved
to permissive licences (see the note above). MMLU, MMLU-Pro and GSM8K were added — they appear
verbatim in released evaluation artifacts and the earlier table omitted them.
