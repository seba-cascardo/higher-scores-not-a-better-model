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

## UNKNOWN — no license declared on the dataset card

**These are declared unknown, not assumed permissive.** Each has items tracked in this
repository. Absence of a stated license is not a grant: if you plan to redistribute
anything derived from these, resolve the license with the upstream source first.

| Dataset | License | What it requires | Met? |
|---|---|---|---|
| **SocialIQA** | **UNKNOWN** — none on card | Undetermined | **Cannot be determined** |
| **HellaSwag** | **UNKNOWN** — none on card | Undetermined | **Cannot be determined** |
| **WinoGrande** | **UNKNOWN** — none on card | Undetermined | **Cannot be determined** |
| **PIQA** | **UNKNOWN** — none on card | Undetermined | **Cannot be determined** |
| **TriviaQA** | **UNKNOWN** — none on card | Undetermined | **Cannot be determined** |
| **PopQA** | **UNKNOWN** — none on card | Undetermined | **Cannot be determined** |
| **`hails/bigbench`** (mirror) | **UNKNOWN** — none on card | Undetermined | **Cannot be determined** |

## GATED — not redistributed, purged from history

These three prohibit redistribution. Their items were **removed from the git history**
with `filter-repo` before this repository was made public, and `.gitignore` names every
file so they cannot be re-adopted. Reproducing results that use them requires obtaining
them from the original source under its own terms.

| Dataset | License | What it requires | Met? |
|---|---|---|---|
| **GPQA-Diamond** | Gated; redistribution prohibited | Obtain from source; do not redistribute items | Yes — 27 files purged, not shipped |
| **EWoK** | Gated; training data may not be redistributed | Obtain from source | Yes — 2 files purged, not shipped |
| **LIMA** | Gated | Obtain from source | Yes — 2 files purged, not shipped |

---

## Not verified (declared, not assumed)

- **If WildChat-1M turns out to be gated** rather than ODC-BY, the 11 files under
  `runs/mixed_class_corpus/` identified by the audit move from "attribution" to
  **blocking**, and would have to be purged the same way the three gated datasets were.
  This was not verified.
- Large binaries hosted on HuggingFace (activation `.pt`, LoRA `.safetensors`) were not
  opened byte by byte; their license verdict rests on the logs that produced them.
- The audit covered the 20 HuggingFace repos relevant to the paper, not all 49 in the
  account.

## Provenance

Built from the license audit in
[`docs/superpowers/specs/2026-08-04-release-remediation-report.md`](docs/superpowers/specs/2026-08-04-release-remediation-report.md)
(§6), which classified artifacts **by content**, not by filename — a distinction that
mattered: files named `kr_*` hold ARC items, a file named `*_c4rerun` holds WikiText, and
the worst GPQA file carried no `Pre-Revision *` columns at all.
