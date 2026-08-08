# Pre-registration protocols

These are the protocol documents for the paper's evaluation arms: the measurements, the
decision thresholds and the kill-rules, each fixed **before** the corresponding run.

**They are the original working documents, reproduced unedited.** That is deliberate. A
pre-registration is only worth what its timestamp is worth, and a document rewritten after
the fact is a different document — it would carry the same filename and a later claim. So
they ship as they were written, including their working language (Spanish), their internal
shorthand, and in two cases a `DRAFT` banner left over from the project stage they were
written in. Nothing in them has been tightened, translated or tidied for release.

Two conventions a reader will hit:

- `[[double-bracket names]]` are links into the project's internal notebook, which is not
  part of this release. They point at working notes, not at evidence; every quantity a
  protocol relies on is in `runs/` or in the paper.
- Paths like `/workspace/…` are the evaluation host's layout, described in the paper's
  methods section. Several documents also carry paste-ready shell for that host.
- References to a "claims ledger", to branch names, and to bracketed identifiers like `B49`
  are the project's internal bookkeeping. The ledger is a private register that records where
  each published quantity was measured; it is not part of this release, and nothing in the
  paper depends on reading it. The artifacts themselves are in `runs/`.
- `ClaraVT` is the project's internal label for an adversarial design-review pass — a
  structured critique run against a proposed experiment before it was executed. It is a
  working procedure, not a person and not a co-author. Where a protocol credits a decision to
  it, the decision is the author's; the label records which review it came out of.

## What each one covers

| Protocol | Date | Covers |
|---|---|---|
| [`2026-06-22-eval-bulletproof-preregistration.md`](2026-06-22-eval-bulletproof-preregistration.md) | 2026-06-22 | The robustness battery behind the six headline numbers, with a `bulletproof-iff / retract-if` condition attached to each one. The parent document for most of what follows. |
| [`2026-06-22-oq1-vg-execution-prereg.md`](2026-06-22-oq1-vg-execution-prereg.md) | 2026-06-22 | The functional-axis measurement (`v_inf` / `v_ret`, whitened angle) and the verifier-guided de-risk arm, with the verdict map fixed in advance. |
| [`2026-06-23-vinf-offset-causal-prereg.md`](2026-06-23-vinf-offset-causal-prereg.md) | 2026-06-23 | The causal test of the offset: is the lift correctness, or an off-axis push? Metrics and construction fixed before the substitution arms were run. |
| [`2026-07-25-e4-e6-preregistration.md`](2026-07-25-e4-e6-preregistration.md) † | 2026-07-25 | E4, the objective × domain factorial that separates *any fine-tuning* from *fine-tuning on this domain*; and E6, the cross-family causal replication on Qwen. Backs the same-parameterization control and the ratio in the README's claim table. |
| [`2026-07-27-e8-scale-coupling-preregistration.md`](2026-07-27-e8-scale-coupling-preregistration.md) | 2026-07-27 | Whether the scoring lift and the length compression are one gain or two, tested by curve shape rather than by a rank correlation that is true by construction. Analysis: `scripts/analyze_scale_coupling.py`. |
| [`2026-07-27-paired-generation-preregistration.md`](2026-07-27-paired-generation-preregistration.md) | 2026-07-27 | The paired generation evaluation: byte-identical items across arms, frozen to JSON before any arm ran. |
| [`2026-07-28-loop-suppression-control-preregistration.md`](2026-07-28-loop-suppression-control-preregistration.md) | 2026-07-28 | Whether the damage to chain-of-thought is emission or reasoning — the control that separates degeneration from lost capability. |

## What the timestamps do and do not establish

Six of the seven entered version control on the date in their name, before the runs they
govern, and that ordering is checkable here:

```bash
git log --diff-filter=A --format='%ad %s' --date=short -- docs/protocols/
```

Two caveats, both worth stating plainly rather than leaving for a reader to discover.

This release was cut from a larger private repository with `git filter-repo`, which
preserves author and commit dates but rewrites commit identifiers. **The dates are the
original ones; the hashes are not.**

**† The E4/E6 protocol is the exception.** It was registered in the private repository on
2026-07-25 (commit `713d4fa`, *"runbook pre-registered"*), ahead of the runs it governs, but
it was not among the paths carried over in the original cut and was added to this release
afterwards. Its add-commit here therefore carries a later date and establishes nothing on
its own. We include it because the alternative is worse: the same-parameterization control
is a headline row, and publishing every protocol except the one behind a headline number
would be the wrong omission. Its date is asserted, not demonstrated.

These are internal protocol documents, not entries in a third-party registry, and the paper
says so where it uses the term.
