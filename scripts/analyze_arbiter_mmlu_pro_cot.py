"""U-16 — the arbiter on MMLU-Pro. LOCAL, CPU, EUR 0.

Same read as scripts/analyze_arbiter_arc_cot.py (the mould): partition the items by
what the ADAPTER'S COLD SCORING did to them, then report the BASE's gen-CoT accuracy
inside each cell. The claim under test is (d): the items the adapter "fixes" in cold
MC scoring were already accessible to the base by reasoning -- i.e. the lift is a
readout repair, not new capability.

Why here and not (only) on ARC: ARC's base sits near its own CoT ceiling, so a high
base-CoT accuracy in the fixed cell is nearly forced. MMLU-Pro has base cold 0.2610
against base CoT 0.7060 -- ~44pp of headroom -- so "the base already solved them"
is falsifiable: the fixed cell could land at the not-fixed cell's rate, or below.

NULL: within the base-cold-WRONG stratum, permute the "adapter fixed it" label.
That preserves the CONDITIONAL law (P(base-CoT correct | base cold wrong)) and the
cell sizes; it does NOT preserve the marginal base-CoT rate over all 1000 items,
which is irrelevant here. Realised exactly by Fisher's exact test on the 2x2
(fixed / not-fixed) x (base-CoT correct / not).

UNPARSED: base CoT has unparsed traces. Per the B57/B58 rule (a parser with a
fallback does not measure its own fallback, and truncated != malformed) this script
does NOT adjudicate them: it reports the canonical scoring (unparsed = wrong) plus
Manski bounds (all unparsed wrong / all right) and the per-cell unparsed counts.

  python scripts/analyze_arbiter_mmlu_pro_cot.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = "runs/paired_gen_dl/mmlu_pro__base.json"
ADPT = "runs/paired_gen_dl/mmlu_pro__v7mc.json"
OUT = "runs/paired_gen_dl/arbiter_mmlu_pro_cot.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def _load(path, expect_arm):
    d = json.load(open(path, encoding="utf-8"))
    arm = d.get("arm")
    if arm != expect_arm:
        raise SystemExit(f"FATAL: {path} has arm={arm!r}, expected {expect_arm!r} "
                         f"(identify an artefact by CONTENT, not by filename).")
    if not d.get("complete"):
        raise SystemExit(f"FATAL: {path} is not marked complete.")
    recs = {r["doc_id"]: r for r in d["records"]}
    if len(recs) != len(d["records"]):
        raise SystemExit(f"FATAL: duplicate doc_id in {path}.")
    return d, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADPT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    db, rb = _load(args.base, "base")
    da, ra = _load(args.adapter, "v7mc")
    print(f"base    : {args.base}\n          items={db.get('items_file')} n={len(rb)} "
          f"chat_template={db.get('apply_chat_template')} max_new={db.get('max_new')}")
    print(f"adapter : {args.adapter}\n          offsets={da.get('offsets')} n={len(ra)} "
          f"chat_template={da.get('apply_chat_template')} max_new={da.get('max_new')}")

    ids = sorted(set(rb) & set(ra))
    if len(ids) != len(rb) or len(ids) != len(ra):
        print(f"  WARNING: doc_id sets differ; using the {len(ids)} in common.")
    bad = [i for i in ids if rb[i]["gold"] != ra[i]["gold"]]
    if bad:
        raise SystemExit(f"FATAL: gold disagrees on {len(bad)} doc_ids — not the same items.")
    print(f"  paired on doc_id: {len(ids)} items; gold agrees on all.\n")

    cells = {"fixed": [], "broken": [], "kept_right": [], "both_wrong": []}
    for i in ids:
        b, a = rb[i]["cold_correct"], ra[i]["cold_correct"]
        key = ("kept_right" if (b and a) else "broken" if (b and not a)
               else "fixed" if (a and not b) else "both_wrong")
        cells[key].append(i)

    n = len(ids)
    b_cold = sum(rb[i]["cold_correct"] for i in ids)
    a_cold = sum(ra[i]["cold_correct"] for i in ids)
    b_cot = sum(rb[i]["cot_correct"] for i in ids)
    b_unp = sum(rb[i]["cot_unparsed"] for i in ids)
    print("=" * 74)
    print(f"MMLU-Pro n={n}   base cold {b_cold/n:.4f}   adapter cold {a_cold/n:.4f}   "
          f"lift {100*(a_cold-b_cold)/n:+.2f}pp")
    print(f"                 base CoT  {b_cot/n:.4f}   (unparsed {b_unp})   "
          f"headroom cold->CoT {100*(b_cot-b_cold)/n:+.2f}pp")
    print("=" * 74)

    rows = {}
    print("\nBASE gen-CoT accuracy inside each cell (unparsed scored WRONG):")
    for name in ["fixed", "broken", "kept_right", "both_wrong"]:
        ids_c = cells[name]
        nc = len(ids_c)
        k = sum(rb[i]["cot_correct"] for i in ids_c)
        u = sum(rb[i]["cot_unparsed"] for i in ids_c)
        lo, hi = wilson(k, nc)
        rows[name] = {"n": nc, "base_cot_correct": k,
                      "acc": (k / nc if nc else float("nan")),
                      "wilson95": [lo, hi], "unparsed": u,
                      "manski_bounds": [k / nc if nc else float("nan"),
                                        (k + u) / nc if nc else float("nan")]}
        print(f"  {name:11s} n={nc:4d}  base-CoT {k:4d}/{nc:<4d} = "
              f"{rows[name]['acc']:.4f}  CI[{lo:.4f},{hi:.4f}]  "
              f"unparsed={u:3d}  Manski[{rows[name]['manski_bounds'][0]:.4f},"
              f"{rows[name]['manski_bounds'][1]:.4f}]")

    # --- PRIMARY: within the base-cold-wrong stratum, fixed vs not-fixed ---------
    f, w = rows["fixed"], rows["both_wrong"]
    kf, nf = f["base_cot_correct"], f["n"]
    kw, nw = w["base_cot_correct"], w["n"]
    from scipy.stats import fisher_exact
    odds, p = fisher_exact([[kf, nf - kf], [kw, nw - kw]])
    diff = kf / nf - kw / nw
    # Newcombe (Wilson) interval for the difference of two independent proportions.
    l1, u1 = wilson(kf, nf)
    l2, u2 = wilson(kw, nw)
    d_lo = diff - ((kf / nf - l1) ** 2 + (u2 - kw / nw) ** 2) ** 0.5
    d_hi = diff + ((u1 - kf / nf) ** 2 + (kw / nw - l2) ** 2) ** 0.5

    print("\n" + "=" * 74)
    print("PRIMARY TEST — base-cold-WRONG stratum only "
          f"(n={nf + nw}); null = permute the 'adapter fixed it' label inside it")
    print(f"  fixed      {kf}/{nf} = {kf/nf:.4f}")
    print(f"  not-fixed  {kw}/{nw} = {kw/nw:.4f}")
    print(f"  difference {100*diff:+.2f}pp   Newcombe95 [{100*d_lo:+.2f},{100*d_hi:+.2f}]pp")
    print(f"  Fisher exact two-sided p = {p:.3e}   OR = {odds:.3f}")
    print("=" * 74)

    # Manski-bounded worst case for the same contrast: unparsed hurt the fixed cell
    # and help the not-fixed cell, and vice versa.
    wc_lo = (kf / nf) - ((kw + w["unparsed"]) / nw)
    wc_hi = ((kf + f["unparsed"]) / nf) - (kw / nw)
    print(f"  worst-case difference under Manski (all unparsed adverse): "
          f"{100*wc_lo:+.2f}pp ; best case {100*wc_hi:+.2f}pp")

    out = {
        "base_file": args.base, "adapter_file": args.adapter,
        "n_paired": n, "base_cold_acc": b_cold / n, "adapter_cold_acc": a_cold / n,
        "cold_lift_pp": 100 * (a_cold - b_cold) / n,
        "base_cot_acc": b_cot / n, "base_cot_unparsed": b_unp,
        "cells": rows,
        "primary": {
            "stratum": "base cold-wrong",
            "n_stratum": nf + nw,
            "fixed": [kf, nf], "not_fixed": [kw, nw],
            "diff_pp": 100 * diff, "newcombe95_pp": [100 * d_lo, 100 * d_hi],
            "fisher_p": p, "odds_ratio": odds,
            "manski_diff_pp": [100 * wc_lo, 100 * wc_hi],
            "null": "permutation of the 'fixed' label inside the base-cold-wrong "
                    "stratum; preserves the CONDITIONAL law and both cell sizes, "
                    "not the marginal base-CoT rate over all items",
        },
        "cell_ids": {k: v for k, v in cells.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
