#!/usr/bin/env python3
"""Persist the full-split arbiter contrast: fixed items vs the negative control.

Why this exists (B64, 2026-08-03)
---------------------------------
The paper's strongest single result -- the arbiter re-run on the whole ARC-Challenge
test split, outside the n=400 draw every other diagnostic shares -- was quoting a
Fisher exact p of 1.4e-8 that lived in NO artifact. The pod script that produced the
cells left the test as a "LOCAL next" step that never ran, so the number was
reproducible in principle and stored nowhere. That is the same shape as the stale
tarball and the spec-only cost benchmark: a load-bearing figure whose only carrier
was prose.

This recomputes it from the raw arbiter files rather than from the cells printed in
the .tex, so the artifact is derived from data, not from the claim it backs.

Usage:
    python scripts/analyze_arbiter_fullsplit_fisher.py
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parent.parent
COT = REPO / "runs/vinf_causal/arbiter_arc_cot_fullsplit.json"
SETS = REPO / "runs/vinf_causal/arbiter_arc_sets_fullsplit.json"
OUT = REPO / "runs/vinf_causal/arbiter_fullsplit_fisher.json"


def main() -> None:
    cot = json.loads(COT.read_text(encoding="utf-8"))
    sets = json.loads(SETS.read_text(encoding="utf-8"))

    correct = {
        r["doc_id"]: bool(r.get("base_cot_correct"))
        for r in cot["results"]
        if "doc_id" in r
    }

    fixed_ids = [d for d in sets["sets"]["mc_only"] if d in correct]
    # The negative control: items NEITHER arm fixes. Same CoT protocol, same run,
    # so the contrast isolates "was it fixed" rather than anything about the prompt.
    both_wrong = sets.get("n_both_wrong")
    ctrl_ids = [
        d for d in (both_wrong if isinstance(both_wrong, list) else [])
        if d in correct
    ]
    if not ctrl_ids:  # stored as a count, not a list: fall back to the complement
        named = set()
        for key in ("mc_only", "fixed_by_both", "vinf_only", "base_right"):
            named.update(sets["sets"].get(key, []))
        ctrl_ids = [d for d in correct if d not in named]

    fixed_ok = sum(correct[d] for d in fixed_ids)
    ctrl_ok = sum(correct[d] for d in ctrl_ids)
    table = [
        [fixed_ok, len(fixed_ids) - fixed_ok],
        [ctrl_ok, len(ctrl_ids) - ctrl_ok],
    ]
    odds, p = fisher_exact(table, alternative="two-sided")

    payload = {
        "source": {"cot": COT.name, "sets": SETS.name},
        "note": (
            "Fisher exact, two-sided, on base CoT correctness: items the adapter "
            "fixes vs items neither arm fixes, on the full ARC-Challenge test split "
            "(outside the shared n=400 draw). Recomputed from the raw arbiter files."
        ),
        "fixed_items": {
            "n": len(fixed_ids),
            "base_cot_correct": fixed_ok,
            "acc": fixed_ok / len(fixed_ids) if fixed_ids else None,
        },
        "negative_control": {
            "n": len(ctrl_ids),
            "base_cot_correct": ctrl_ok,
            "acc": ctrl_ok / len(ctrl_ids) if ctrl_ids else None,
        },
        "contingency_table": table,
        "odds_ratio": odds,
        "p_value": p,
        "n_unparsed_in_cot_run": cot["summary"].get("n_unparsed"),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"fixed items      : {fixed_ok}/{len(fixed_ids)} = {payload['fixed_items']['acc']:.4f}")
    print(f"negative control : {ctrl_ok}/{len(ctrl_ids)} = {payload['negative_control']['acc']:.4f}")
    print(f"Fisher OR        : {odds:.4f}")
    print(f"p                : {p:.4g}")
    print(f"unparsed in run  : {payload['n_unparsed_in_cot_run']}")
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
