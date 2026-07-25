"""FASE 1b -- is the heterogeneity the DISTRIBUTION, not the push?

FASE 1 killed the distractor hypothesis on a specific fact: the plain-CE margin is
FLAT across tasks (Gemma +3.15 ARC / +3.65 TQA / +3.67 HSwag) while the inducibility
it is supposed to explain runs 0.9% -> 36.2% -> 69.5%. A constant regressor cannot
explain a 70x-varying response.

What varies instead may be the TARGET, not the push: how many items sit near the
decision boundary. A fixed-size shove flips many items where the base gap is packed
around zero and few where it is not. The project already has a hint from another
direction -- B10d/Test1 measured that on ARC the base sits ON THE EDGE (latent margin
-1.26, 149 flips of 400).

Per item: gap_base = logp(gold)_base - logp(best distractor)_base, sign = correctness
under raw acc. The margin measured in FASE 1 is the shift applied to that gap.

The falsifiable version -- a UNIFORM-PUSH counterfactual. Take the arm's MEAN margin,
apply it to every item's base gap, and predict the accuracy change:

  d_acc_predicted = mean[gap_base + margin_mean > 0] - mean[gap_base > 0]

Compare to the measured d_acc. This separates two worlds:
  * predicted ~= measured  -> the heterogeneity IS the base gap distribution; the arm
                              applies an undifferentiated shove and geometry does the
                              rest. That would be a closed mechanism.
  * predicted != measured  -> the arm is SELECTIVE (it pushes the items that matter),
                              and the residual is where the real mechanism lives.

Predictions fixed BEFORE running:
  Q1  near-zero density of gap_base orders ARC > HSwag > TQA in Gemma (the order of
      the measured raw-acc effects: +9.4 / +2.0 / +0.8 pp)
  Q2  the uniform-push counterfactual reproduces measured d_acc within +-3pp in most
      cells
  Q3  if Q2 fails, the SIGN of the residual says whether arms are selective

Every cross-cell claim carries its sigma; under 2 sigma is not claimed.

  python scripts/analyze_threshold_density_mechanism.py
"""
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

E1, E4, E6 = "runs/e1_plain_ce", "runs/e4_factorial", "runs/e6_qwen_spine"
OUT = "runs/e4_factorial/threshold_density_mechanism.json"
TASKS = ["arc_challenge", "hellaswag", "truthfulqa_mc1"]
# near-zero band, in nats: the scale of the measured plain-CE margins (~3)
BANDS = [1.0, 3.0, 10.0]

ARMS = [
    ("Gemma contrastive canonical", "Gemma", "contrastive", f"{E1}/eval_base.json",
     [f"{E1}/eval_contrastive_canonical.json"]),
    ("Gemma plain-CE in-domain", "Gemma", "plain-CE", f"{E1}/eval_base.json",
     [f"{E1}/eval_lifts_seed{s}.json" for s in (0, 1, 2)]),
    ("Gemma plain-CE OOD", "Gemma", "plain-CE", f"{E1}/eval_base.json",
     [f"{E4}/eval_ood_plain_ce_seed{s}.json" for s in (0, 1, 2)]),
    ("Gemma contrastive OOD", "Gemma", "contrastive", f"{E1}/eval_base.json",
     [f"{E4}/eval_ood_direct_seed{s}.json" for s in (0, 1, 2)]),
    ("Qwen contrastive", "Qwen", "contrastive", f"{E6}/eval_base.json",
     [f"{E6}/eval_mc.json"]),
    ("Qwen plain-CE", "Qwen", "plain-CE", f"{E6}/eval_base.json",
     [f"{E6}/eval_plain_ce_seed{s}.final.json" for s in (0, 1, 2)]),
]


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


def gaps(doc):
    """gap = logp(gold) - logp(best distractor), per item, in stable key order."""
    keys = sorted(doc)
    g = []
    for k in keys:
        lp, tgt = doc[k]
        others = [i for i in range(len(lp)) if i != tgt]
        if not others:
            continue
        g.append(lp[tgt] - max(lp[i] for i in others))
    return np.array(g), keys


def cell(base_doc, arm_doc):
    """Measured shift vs the uniform-push counterfactual, on shared items."""
    keys = sorted(set(base_doc) & set(arm_doc))
    gb, ga = [], []
    for k in keys:
        b_lp, tgt = base_doc[k]
        a_lp, tgt_a = arm_doc[k]
        if tgt != tgt_a or len(b_lp) != len(a_lp) or len(b_lp) < 2:
            continue
        o = [i for i in range(len(b_lp)) if i != tgt]
        gb.append(b_lp[tgt] - max(b_lp[i] for i in o))
        ga.append(a_lp[tgt] - max(a_lp[i] for i in o))
    gb, ga = np.array(gb), np.array(ga)
    if not len(gb):
        return None
    shift = ga - gb                       # per-item margin (arm's own top distractor)
    m = float(shift.mean())
    acc_b, acc_a = float((gb > 0).mean()), float((ga > 0).mean())
    pred = float((gb + m > 0).mean())     # uniform push of the SAME mean size
    # corr(shift, gap_base) is NOT a selectivity measure: shift = ga - gb contains gb
    # with a negative sign, so any compression of the distribution drives it toward -1
    # (regression to the mean). Kept for the record, never read as selectivity.
    sel_naive = float(np.corrcoef(shift, gb)[0, 1]) if shift.std() > 0 else 0.0

    # ------------------------------------------------------------------------
    # WARNING -- the GLOBAL permutation below is NOT a valid selectivity control,
    # and the "selectivity" result this script first reported was an artifact of
    # using it. Refuted by adversarial verification 2026-07-25.
    #
    # Permuting shifts across ALL items destroys the dependence of shift on
    # gap_base -- but that dependence exists for a reason that has nothing to do
    # with aiming: the arms RESCALE the option scores (plain-CE compress the gap
    # spread 2.5-4x, contrastive arms expand it 1.5-2.7x). Any rescaling makes
    # shift = (a-1)*gap_base + c, anticorrelated with gap_base BY CONSTRUCTION and
    # with no knowledge of which option is correct. So the global permutation
    # rejects for every arm that rescales, which is all of them.
    #
    # Measured against the CORRECT null (shift = f(gap_base) + exchangeable noise,
    # i.e. permute WITHIN strata of gap_base): correlation with the observed
    # residual is +0.979 and only 2/18 cells clear 2 sigma, versus 9/18 under the
    # global permutation. The star case falls from 7.1 sigma to z = +0.15.
    #
    # Both are computed. `residual_vs_permuted_pp` is kept for the record and must
    # NOT be read as selectivity; `residual_vs_stratified_pp` is the honest one.
    # ------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    perm = np.array([float(((gb + rng.permutation(shift)) > 0).mean()) for _ in range(200)])

    # Stratified null: bin by gap_base, permute shifts only within a bin. Preserves
    # the whole conditional law of the push given the base gap (mean AND variance),
    # destroys only item identity inside the bin.
    order = np.argsort(gb)
    n_bins = max(4, len(gb) // 25)
    strata = np.array_split(order, n_bins)
    strat = []
    for _ in range(200):
        sh = shift.copy()
        for idx in strata:
            sh[idx] = rng.permutation(shift[idx])
        strat.append(float(((gb + sh) > 0).mean()))
    strat = np.array(strat)

    # The item-blind replacement: fit a single affine map ga ~ a*gb + c and see how
    # much of the accuracy change two global scalars reproduce with no noise at all.
    a_hat, c_hat = np.polyfit(gb, ga, 1)
    acc_affine = float(((a_hat * gb + c_hat) > 0).mean())
    return {"n": len(gb), "margin_mean": m,
            "margin_sd_items": float(shift.std(ddof=1)),
            "gap_base_sd": float(gb.std(ddof=1)),
            "gap_arm_sd": float(ga.std(ddof=1)),
            "spread_ratio_arm_over_base": float(ga.std(ddof=1) / gb.std(ddof=1)),
            "selectivity_corr_naive_DO_NOT_READ": sel_naive,
            "acc_base": acc_b, "acc_arm": acc_a,
            "d_acc_measured_pp": (acc_a - acc_b) * 100,
            "d_acc_uniform_pp": (pred - acc_b) * 100,
            # --- the item-blind affine model: two global scalars, no noise ---
            "affine_slope": float(a_hat), "affine_intercept": float(c_hat),
            "effective_threshold_shift": float(-c_hat / a_hat) if a_hat else None,
            "d_acc_affine_pp": (acc_affine - acc_b) * 100,
            "affine_frac_of_measured": (float((acc_affine - acc_b) / (acc_a - acc_b))
                                        if acc_a != acc_b else None),
            # --- the invalid control, kept only for the record ---
            "d_acc_permuted_pp": float((perm.mean() - acc_b) * 100),
            "residual_vs_permuted_pp_DO_NOT_READ": float((acc_a - perm.mean()) * 100),
            "sigma_vs_permuted_DO_NOT_READ": (float(abs(acc_a - perm.mean()) / perm.std(ddof=1))
                                              if perm.std(ddof=1) > 0 else float("inf")),
            # --- the honest control: permute within strata of gap_base ---
            "d_acc_stratified_pp": float((strat.mean() - acc_b) * 100),
            "residual_vs_stratified_pp": float((acc_a - strat.mean()) * 100),
            "sigma_vs_stratified": (float(abs(acc_a - strat.mean()) / strat.std(ddof=1))
                                    if strat.std(ddof=1) > 0 else float("inf")),
            "residual_pp": (acc_a - pred) * 100}


def main():
    print(f"{'=' * 100}\nFASE 1b -- base-gap density and the uniform-push counterfactual\n{'=' * 100}")

    # --- Q1: how packed is the base gap around zero, per task and family? ------
    print("\nQ1 -- base-gap density near the decision boundary (|gap| < band, nats)")
    print(f"  {'family':<8}{'task':<16}{'n':>5}{'median|gap|':>13}" +
          "".join(f"{'<' + str(b):>9}" for b in BANDS) + f"{'acc_base':>10}")
    density = {}
    for fam, bp in (("Gemma", f"{E1}/eval_base.json"), ("Qwen", f"{E6}/eval_base.json")):
        base = load(bp)
        for t in TASKS:
            if t not in base:
                continue
            g, _ = gaps(base[t])
            row = {"n": len(g), "median_abs_gap": float(np.median(np.abs(g))),
                   "acc_base": float((g > 0).mean()),
                   "frac_within": {str(b): float((np.abs(g) < b).mean()) for b in BANDS}}
            density[f"{fam}/{t}"] = row
            print(f"  {fam:<8}{t:<16}{len(g):>5}{row['median_abs_gap']:>13.2f}" +
                  "".join(f"{row['frac_within'][str(b)] * 100:>8.1f}%" for b in BANDS) +
                  f"{row['acc_base']:>10.4f}")

    # --- Q2/Q3: uniform-push counterfactual per arm ---------------------------
    print(f"\nQ2/Q3 -- does an undifferentiated push of the measured size reproduce the effect?")
    print(f"  {'arm':<28}{'task':<16}{'measured':>10}{'spread':>8}{'thr shift':>10}"
          f"{'affine':>9}{'aff %':>8}{'strat':>9}{'σ strat':>9}")
    cells, bases = {}, {}
    for label, fam, obj, bp, arms in ARMS:
        if bp not in bases:
            bases[bp] = load(bp)
        base = bases[bp]
        per_task = {}
        for t in TASKS:
            seeds = [cell(base[t], load(p)[t]) for p in arms
                     if os.path.exists(p) and t in base and t in load(p)]
            seeds = [s for s in seeds if s]
            if not seeds:
                continue

            def across(k):
                # a cell whose measured effect is exactly zero has no defined
                # "fraction of measured"; drop those instead of dividing by it
                v = np.array([s[k] for s in seeds if s[k] is not None], dtype=float)
                if not len(v):
                    return {"mean": None, "sd": None}
                return {"mean": float(v.mean()),
                        "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

            r = {k: across(k) for k in ("margin_mean", "margin_sd_items", "gap_base_sd",
                                        "spread_ratio_arm_over_base", "affine_slope",
                                        "affine_intercept", "effective_threshold_shift",
                                        "d_acc_measured_pp", "d_acc_uniform_pp",
                                        "d_acc_affine_pp", "affine_frac_of_measured",
                                        "d_acc_permuted_pp", "residual_vs_permuted_pp_DO_NOT_READ",
                                        "sigma_vs_permuted_DO_NOT_READ",
                                        "d_acc_stratified_pp", "residual_vs_stratified_pp",
                                        "sigma_vs_stratified", "residual_pp")}
            r["n_seeds"], r["n"] = len(seeds), seeds[0]["n"]
            per_task[t] = r
            af = r["affine_frac_of_measured"]["mean"]
            afs = f"{af * 100:>7.0f}%" if af is not None else "      -" 
            print(f"  {label:<28}{t:<16}{r['d_acc_measured_pp']['mean']:>+10.2f}"
                  f"{r['spread_ratio_arm_over_base']['mean']:>8.2f}"
                  f"{r['effective_threshold_shift']['mean']:>+10.2f}"
                  f"{r['d_acc_affine_pp']['mean']:>+9.2f}{afs}"
                  f"{r['d_acc_stratified_pp']['mean']:>+9.2f}"
                  f"{r['sigma_vs_stratified']['mean']:>9.2f}", flush=True)
        cells[label] = {"family": fam, "objective": obj, "per_task": per_task}

    # --- verdicts -------------------------------------------------------------
    print(f"\n{'=' * 100}\nVERDICTS\n{'=' * 100}")
    g_order = [(t, density[f"Gemma/{t}"]["frac_within"]["3.0"]) for t in TASKS
               if f"Gemma/{t}" in density]
    g_order.sort(key=lambda x: -x[1])
    print(f"  Q1  Gemma near-zero density order (|gap|<3): "
          f"{' > '.join(f'{t.split(chr(95))[0]} {v * 100:.1f}%' for t, v in g_order)}")
    print(f"      measured raw-acc effect order was ARC +9.4 > HSwag +2.0 > TQA +0.8 pp")

    resid = [(l, t, c["per_task"][t]["residual_pp"]) for l, c in cells.items()
             for t in c["per_task"]]
    within3 = [r for r in resid if abs(r[2]["mean"]) <= 3.0]
    print(f"  Q2  uniform push within ±3pp of measured in {len(within3)}/{len(resid)} cells")

    perm = [(l, t, c["per_task"][t]["residual_vs_stratified_pp"],
             c["per_task"][t]["sigma_vs_stratified"]) for l, c in cells.items()
            for t in c["per_task"]]
    close = [r for r in perm if r[3]["mean"] < 2.0]
    print(f"  Q3  STRATIFIED null (permute shifts WITHIN bins of gap_base -- preserves the"
          f" whole conditional law of the push given the base gap):")
    print(f"      measured indistinguishable from the null in {len(close)}/{len(perm)} cells")
    aff = [c["per_task"][t]["affine_frac_of_measured"]["mean"] for l, c in cells.items()
           for t in c["per_task"] if c["per_task"][t]["affine_frac_of_measured"]["mean"] is not None]
    print(f"      item-blind affine map reproduces a median {np.median(aff) * 100:.0f}% of the"
          f" measured accuracy change (two global scalars, no noise, no item information)")
    for l, t, r, sg in sorted(perm, key=lambda r: -r[3]["mean"])[:6]:
        tag = "survives" if sg["mean"] >= 2.0 else "explained by rescaling"
        print(f"        {l:<28}{t:<16}{r['mean']:>+8.2f}pp  ({sg['mean']:>5.2f}σ)  {tag}")

    out = {"predictions_fixed_before_running": {
        "Q1": "near-zero base-gap density orders ARC > HSwag > TQA in Gemma",
        "Q2": "uniform-push counterfactual reproduces measured d_acc within ±3pp in most cells",
        "Q3": "if Q2 fails, the sign of the residual says whether arms are selective"},
        "definitions": {"gap": "logp(gold) - logp(best distractor), raw log-probs",
                        "uniform_push": "every item's gap shifted by the arm's MEAN margin",
                        "residual": "measured d_acc - uniform-push d_acc"},
        "base_gap_density": density, "cells": cells,
        "q1_gemma_density_order": [[t, v] for t, v in g_order],
        "q2_cells_within_3pp": [len(within3), len(resid)]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
