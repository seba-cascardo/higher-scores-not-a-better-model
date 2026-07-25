"""Is the one cell that survives the stratified null real, or is it multiplicity?

FASE 1b's corrected analysis ran a stratified permutation null over 18 cells and
found exactly one that clears 2 sigma: Gemma contrastive canonical / HellaSwag,
+1.27pp at 2.90 sigma. That number cannot be read as-is:

  * 200 permutations do not pin down a tail at ~3 sigma.
  * 18 cells were tested. Bonferroni at 5% needs z >= 3.0 here, and 2.90 does not
    clear it -- but under the null you would expect 0.07 cells at that level, not
    1, so "just noise" is not established either.
  * sigma = |measured - mean(null)| / sd(null) treats the MEASURED value as exact.
    It is one eval of 400 items and carries its own sampling error.
  * the stratified null has a free parameter -- the bin width -- and nobody checked
    whether the result depends on it.

This closes all four. Everything is recomputed for all 18 cells so the multiplicity
correction is honest.

  python scripts/audit_stratified_null_survivor.py
"""
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

E1, E4, E6 = "runs/e1_plain_ce", "runs/e4_factorial", "runs/e6_qwen_spine"
OUT = "runs/e4_factorial/stratified_null_survivor_audit.json"
TASKS = ["arc_challenge", "hellaswag", "truthfulqa_mc1"]
N_PERM = 5000              # vs 200 in the original -- a 3-sigma tail needs the reps
BIN_SIZES = [10, 25, 50, 100]   # items per stratum; 25 was the original
N_BOOT = 2000              # paired item bootstrap for the measured value's own error
K_INNER = 40               # permutations per bootstrap replicate

ARMS = [
    ("Gemma contrastive canonical", f"{E1}/eval_base.json", f"{E1}/eval_contrastive_canonical.json"),
    ("Gemma plain-CE in-domain", f"{E1}/eval_base.json", f"{E1}/eval_lifts_seed0.json"),
    ("Gemma plain-CE OOD", f"{E1}/eval_base.json", f"{E4}/eval_ood_plain_ce_seed0.json"),
    ("Gemma contrastive OOD", f"{E1}/eval_base.json", f"{E4}/eval_ood_direct_seed0.json"),
    ("Qwen contrastive", f"{E6}/eval_base.json", f"{E6}/eval_mc.json"),
    ("Qwen plain-CE", f"{E6}/eval_base.json", f"{E6}/eval_plain_ce_seed0.final.json"),
]
FOCUS = ("Gemma contrastive canonical", "hellaswag")


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for t in TASKS:
        if t not in d.get("samples", {}):
            continue
        per = {}
        for s in d["samples"][t]:
            lp = np.array([float(r[0]) for r in s["filtered_resps"]])
            tgt = s["target"]
            if isinstance(tgt, str) or not (0 <= int(tgt) < len(lp)):
                continue
            per[s["doc_hash"]] = (lp, int(tgt))
        if per:
            out[t] = per
    return out


def gaps(base_doc, arm_doc):
    gb, ga = [], []
    for k in sorted(set(base_doc) & set(arm_doc)):
        b_lp, tgt = base_doc[k]
        a_lp, tgt_a = arm_doc[k]
        if tgt != tgt_a or len(b_lp) != len(a_lp) or len(b_lp) < 2:
            continue
        o = [i for i in range(len(b_lp)) if i != tgt]
        gb.append(b_lp[tgt] - max(b_lp[i] for i in o))
        ga.append(a_lp[tgt] - max(a_lp[i] for i in o))
    return np.array(gb), np.array(ga)


def strat_null(gb, shift, per_bin, n_perm, rng):
    """Accuracy under shifts permuted WITHIN strata of gap_base, vectorised.

    Sorting by (block_id * BIG + random) permutes inside each contiguous block of
    the gap-ordered array, which is exactly a within-stratum shuffle.
    """
    n = len(gb)
    order = np.argsort(gb)
    block = np.arange(n) // max(1, per_bin)
    keys = block[None, :] * 10.0 + rng.random((n_perm, n))
    inner = np.argsort(keys, axis=1)                 # (n_perm, n) positions in sorted space
    sh_sorted = shift[order]
    permuted = np.empty((n_perm, n))
    permuted[:, order] = sh_sorted[inner]
    return ((gb[None, :] + permuted) > 0).mean(axis=1)


def main():
    rng = np.random.default_rng(20260725)
    cells, bases = {}, {}
    print(f"{'=' * 96}\nStratified-null audit: {N_PERM} permutations, "
          f"bin sweep {BIN_SIZES}, {N_BOOT} bootstrap reps\n{'=' * 96}")
    print(f"  {'arm':<28}{'task':<12}{'resid pp':>10}" +
          "".join(f"{'z@' + str(b):>9}" for b in BIN_SIZES))

    for label, bp, ap in ARMS:
        for p in (bp, ap):
            if p not in bases:
                bases[p] = load(p)
        for t in TASKS:
            if t not in bases[bp] or t not in bases[ap]:
                continue
            gb, ga = gaps(bases[bp][t], bases[ap][t])
            if not len(gb):
                continue
            shift = ga - gb
            acc_a = float((ga > 0).mean())
            row = {"n": len(gb), "acc_base": float((gb > 0).mean()), "acc_arm": acc_a,
                   "by_bin": {}}
            for b in BIN_SIZES:
                null = strat_null(gb, shift, b, N_PERM, rng)
                resid = (acc_a - null.mean()) * 100
                z = abs(acc_a - null.mean()) / null.std(ddof=1) if null.std(ddof=1) else np.inf
                # one-sided empirical p, the only thing the reps actually support
                more = int((np.abs(null - null.mean()) >= abs(acc_a - null.mean())).sum())
                row["by_bin"][str(b)] = {"residual_pp": float(resid), "z": float(z),
                                         "p_empirical": (more + 1) / (N_PERM + 1),
                                         "null_sd_pp": float(null.std(ddof=1) * 100)}
            cells[f"{label} | {t}"] = row
            zz = "".join(f"{row['by_bin'][str(b)]['z']:>9.2f}" for b in BIN_SIZES)
            print(f"  {label:<28}{t[:11]:<12}"
                  f"{row['by_bin']['25']['residual_pp']:>+10.2f}{zz}", flush=True)

    # ---- multiplicity over the 18 cells, at the original bin size ----------
    print(f"\n{'=' * 96}\nMULTIPLICITY (18 cells, bin=25)\n{'=' * 96}")
    ps = {k: v["by_bin"]["25"]["p_empirical"] for k, v in cells.items()}
    m = len(ps)
    order = sorted(ps, key=lambda k: ps[k])
    bonf = 0.05 / m
    print(f"  Bonferroni threshold at 5%: p <= {bonf:.5f}   (m={m})")
    surv_bonf, surv_bh = [], []
    for i, k in enumerate(order, 1):
        bh = 0.05 * i / m
        tag = []
        if ps[k] <= bonf:
            tag.append("BONFERRONI")
            surv_bonf.append(k)
        if ps[k] <= bh:
            tag.append("BH-FDR")
            surv_bh.append(k)
        if i <= 4:
            print(f"    {i:>2}. {k:<48} p={ps[k]:.5f}  (BH cutoff {bh:.5f})"
                  f"  {' + '.join(tag) if tag else '--'}")
    print(f"  -> survives Bonferroni: {len(surv_bonf)}/{m}   survives BH-FDR: {len(surv_bh)}/{m}")

    # ---- the focus cell: does the MEASURED value's own error kill it? ------
    label, task = FOCUS
    bp = next(b for l, b, _ in ARMS if l == label)
    ap = next(a for l, _, a in ARMS if l == label)
    gb, ga = gaps(bases[bp][task], bases[ap][task])
    shift = ga - gb
    n = len(gb)
    print(f"\n{'=' * 96}\nFOCUS CELL: {label} / {task}\n{'=' * 96}")
    print(f"  bin-size sensitivity: " +
          "  ".join(f"{b}->{cells[f'{label} | {task}']['by_bin'][str(b)]['z']:.2f}σ"
                    for b in BIN_SIZES))

    # paired item bootstrap: resample items, recompute BOTH the measured accuracy
    # and its stratified null on the resampled set, so the measured value stops
    # being treated as exact.
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        gbb, shb = gb[idx], shift[idx]
        acc_b = float(((gbb + shb) > 0).mean())
        null_b = strat_null(gbb, shb, 25, K_INNER, rng)
        boot[i] = (acc_b - null_b.mean()) * 100
        if (i + 1) % 500 == 0:
            print(f"    [{i + 1}/{N_BOOT}] bootstrap", flush=True)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    crosses = bool(lo <= 0 <= hi)
    print(f"  residual with the measured value's OWN error: {boot.mean():+.2f} pp, "
          f"CI95 [{lo:+.2f}, {hi:+.2f}]  -> {'CROSSES ZERO' if crosses else 'excludes zero'}")

    # Three independent criteria, reported separately. Collapsing them into one
    # sentence is how the previous version ended up asserting the CI crossed zero
    # when it does not.
    focus = cells[f"{label} | {task}"]["by_bin"]
    zs = [focus[str(b)]["z"] for b in BIN_SIZES]
    bin_stable = (min(zs) >= 2.0)
    passes_mult = f"{label} | {task}" in surv_bonf
    effect_pp = (cells[f"{label} | {task}"]["acc_arm"] -
                 cells[f"{label} | {task}"]["acc_base"]) * 100
    share = abs(boot.mean()) / abs(effect_pp) * 100 if effect_pp else float("nan")
    if passes_mult and bin_stable and not crosses:
        verdict = ("SURVIVES on all three criteria -- multiplicity, bin stability, and "
                   "the measured value's own error")
    else:
        why = []
        if not passes_mult:
            why.append(f"it fails Bonferroni over the {m} cells it was selected from "
                       f"(p={ps[f'{label} | {task}']:.4f} vs {bonf:.5f})")
        if not bin_stable:
            why.append(f"its significance depends on a free parameter -- z ranges "
                       f"{min(zs):.2f}-{max(zs):.2f} across bin widths {BIN_SIZES}")
        if crosses:
            why.append("its CI crosses zero once the measured value's sampling error counts")
        else:
            why.append(f"the residual does differ from zero ({boot.mean():+.2f} pp, "
                       f"CI [{lo:+.2f}, {hi:+.2f}]) but is only {share:.0f}% of that "
                       f"cell's {effect_pp:+.1f} pp effect")
        verdict = "NOT CLAIMED -- " + "; ".join(why)
    print(f"\n  -> {verdict}")

    out = {"n_permutations": N_PERM, "bin_sizes": BIN_SIZES, "n_bootstrap": N_BOOT,
           "cells": cells, "multiplicity": {"m": m, "bonferroni_threshold": bonf,
                                            "survives_bonferroni": surv_bonf,
                                            "survives_bh_fdr": surv_bh},
           "focus_cell": {"cell": f"{label} | {task}", "bootstrap_mean_pp": float(boot.mean()),
                          "bootstrap_ci95_pp": [float(lo), float(hi)],
                          "ci_crosses_zero": crosses},
           "verdict": verdict}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
