"""R1 (audit remediation) -- is the affine margin map a FIT or a MECHANISM?

The published §5.9 claim rests on fitting ga ~ a*gb + c per cell and reporting how
much of the accuracy change two global scalars reproduce. That fit is IN-SAMPLE:
`analyze_threshold_density_mechanism.py:167-168` calls np.polyfit on (gb, ga) and then
evaluates the reproduced accuracy on the very same gb. An external audit flagged it.

This script answers the only question that matters: do the two scalars fitted on one
half of the items reproduce the accuracy change on the OTHER half?

  gap = logp(gold) - logp(best distractor), per item, exactly as in the canonical script
  split items 50/50 -> fit (a, c) on A -> evaluate on B:
      acc_affine_B = mean[(a*gb_B + c) > 0]
      frac_heldout = (acc_affine_B - acc_base_B) / (acc_arm_B - acc_base_B)

CRITERION FIXED BEFORE RUNNING (see plans/2026-07-27-external-audit-remediation.md):
  PASS  if the cross-cell MEDIAN of the held-out frac lands inside [0.70, 1.93] -- the
        cellwise range the paper already publishes in-sample -- AND >= 14/18 cells keep
        the sign of the measured effect.
  FAIL  if the median leaves that range, or the sign is lost in more than 4 cells.

A FAIL demotes §5.9 from "the mechanism" to "a descriptive affine signature".

Own prediction, registered before running: PASSES comfortably. Two parameters over
n~400 should barely overfit. If it fails, the finding was never the fit.

  python scripts/analyze_affine_map_heldout.py
"""
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

E1, E4, E6 = "runs/e1_plain_ce", "runs/e4_factorial", "runs/e6_qwen_spine"
OUT = "runs/e4_factorial/affine_map_heldout.json"
TASKS = ["arc_challenge", "hellaswag", "truthfulqa_mc1"]
N_SPLITS = 200
SEED = 0

# identical to the canonical script, so the cells line up one-to-one
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


def gap_pair(base_doc, arm_doc):
    """(gb, ga) on shared items -- byte-identical logic to the canonical script."""
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
    return np.array(gb), np.array(ga)


def heldout(gb, ga, n_splits=N_SPLITS, seed=SEED):
    """Fit (a, c) on half the items, score the reproduced accuracy on the other half."""
    rng = np.random.default_rng(seed)
    n = len(gb)
    # in-sample reference, recomputed here so the comparison is like-for-like
    a_in, c_in = np.polyfit(gb, ga, 1)
    acc_b_all, acc_a_all = float((gb > 0).mean()), float((ga > 0).mean())
    acc_aff_in = float(((a_in * gb + c_in) > 0).mean())
    frac_in = ((acc_aff_in - acc_b_all) / (acc_a_all - acc_b_all)
               if acc_a_all != acc_b_all else None)

    fracs, d_affine, slopes, intercepts = [], [], [], []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        A, B = idx[: n // 2], idx[n // 2:]
        if len(A) < 10 or len(B) < 10:
            continue
        a_hat, c_hat = np.polyfit(gb[A], ga[A], 1)          # fit on A
        gb_B, ga_B = gb[B], ga[B]                            # evaluate on B
        acc_b = float((gb_B > 0).mean())
        acc_a = float((ga_B > 0).mean())
        acc_aff = float(((a_hat * gb_B + c_hat) > 0).mean())
        d_affine.append((acc_aff - acc_b) * 100)
        slopes.append(float(a_hat))
        intercepts.append(float(c_hat))
        if acc_a != acc_b:
            fracs.append((acc_aff - acc_b) / (acc_a - acc_b))
    if not fracs:
        return None
    f = np.array(fracs)
    return {
        "n_items": n, "n_splits_used": len(fracs),
        "affine_frac_insample": frac_in,
        "affine_frac_heldout_median": float(np.median(f)),
        "affine_frac_heldout_mean": float(f.mean()),
        "affine_frac_heldout_ci": [float(np.percentile(f, 2.5)),
                                   float(np.percentile(f, 97.5))],
        "d_acc_affine_heldout_pp_median": float(np.median(d_affine)),
        "d_acc_measured_pp": (float((ga > 0).mean()) - float((gb > 0).mean())) * 100,
        "slope_insample": float(a_in), "intercept_insample": float(c_in),
        "slope_heldout_sd": float(np.std(slopes, ddof=1)),
        "intercept_heldout_sd": float(np.std(intercepts, ddof=1)),
        "sign_kept": bool(np.sign(np.median(d_affine))
                          == np.sign((float((ga > 0).mean()) - float((gb > 0).mean()))))
        if float((ga > 0).mean()) != float((gb > 0).mean()) else None,
    }


def main():
    print("=" * 104)
    print("R1 -- held-out validation of the affine margin map (fit on half, score on the other half)")
    print("=" * 104)
    print(f"  {'arm':<28}{'task':<16}{'measured':>10}{'in-samp':>9}{'held-out':>10}"
          f"{'CI95':>18}{'sign':>7}")

    bases, cells, flat = {}, {}, []
    total = sum(len(a[4]) for a in ARMS) * len(TASKS)
    done = 0
    for label, fam, obj, bp, arms in ARMS:
        if bp not in bases:
            bases[bp] = load(bp)
        base = bases[bp]
        per_task = {}
        for t in TASKS:
            seeds = []
            for p in arms:
                done += 1
                if not (os.path.exists(p) and t in base):
                    continue
                arm = load(p)
                if t not in arm:
                    continue
                gb, ga = gap_pair(base[t], arm[t])
                if not len(gb):
                    continue
                r = heldout(gb, ga)
                if r:
                    seeds.append(r)
                print(f"    [{done}/{total}] {label} | {t} | {os.path.basename(p)}",
                      flush=True)
            if not seeds:
                continue

            def across(k):
                v = [s[k] for s in seeds if s.get(k) is not None]
                if not v:
                    return None
                return {"mean": float(np.mean(v)),
                        "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}

            agg = {k: across(k) for k in
                   ("affine_frac_insample", "affine_frac_heldout_median",
                    "d_acc_affine_heldout_pp_median", "d_acc_measured_pp",
                    "slope_insample", "intercept_insample",
                    "slope_heldout_sd", "intercept_heldout_sd")}
            lo = float(np.mean([s["affine_frac_heldout_ci"][0] for s in seeds]))
            hi = float(np.mean([s["affine_frac_heldout_ci"][1] for s in seeds]))
            agg["affine_frac_heldout_ci"] = [lo, hi]
            agg["n_seeds"], agg["n_items"] = len(seeds), seeds[0]["n_items"]
            signs = [s["sign_kept"] for s in seeds if s["sign_kept"] is not None]
            agg["sign_kept"] = bool(all(signs)) if signs else None
            per_task[t] = agg

            fi = agg["affine_frac_insample"]
            fh = agg["affine_frac_heldout_median"]
            sg = {True: "ok", False: "LOST", None: "zero"}[agg["sign_kept"]]
            print(f"  {label:<28}{t:<16}{agg['d_acc_measured_pp']['mean']:>+10.2f}"
                  f"{(fi['mean'] if fi else float('nan')):>9.2f}"
                  f"{(fh['mean'] if fh else float('nan')):>10.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>18}{sg:>7}", flush=True)
            if fh:
                flat.append((label, t, fh["mean"], agg["sign_kept"]))
        cells[label] = {"family": fam, "objective": obj, "per_task": per_task}

    # --- verdict ----------------------------------------------------------------
    fr = np.array([f[2] for f in flat])
    med = float(np.median(fr))
    n_sign_ok = sum(1 for f in flat if f[3] is not False)
    in_range = 0.70 <= med <= 1.93
    passed = bool(in_range and n_sign_ok >= 14)

    print("\n" + "=" * 104)
    print("VERDICT (criterion fixed before running)")
    print("=" * 104)
    print(f"  cells evaluated                      : {len(flat)}")
    print(f"  cross-cell MEDIAN held-out frac      : {med:.3f}   "
          f"(criterion: inside [0.70, 1.93] -> {'PASS' if in_range else 'FAIL'})")
    print(f"  cellwise held-out range              : {fr.min():.2f} - {fr.max():.2f}")
    print(f"  cells keeping the measured sign      : {n_sign_ok}/{len(flat)}   "
          f"(criterion: >= 14 -> {'PASS' if n_sign_ok >= 14 else 'FAIL'})")
    print(f"\n  >>> {'PASS' if passed else 'FAIL'} <<<")
    if passed:
        print("  The two scalars generalise out-of-sample. §5.9 stands; the audit's")
        print("  in-sample objection is answered with a number, not a caveat.")
    else:
        print("  The affine map does NOT generalise. §5.9 must be demoted from")
        print("  'the mechanism' to 'a descriptive affine signature'.")

    out = {"criterion_fixed_before_running": {
        "pass": "cross-cell median held-out frac inside [0.70, 1.93] AND >= 14 cells keep sign",
        "fail": "median outside the range, or sign lost in more than 4 cells",
        "rationale": "the in-sample cellwise range the paper already publishes"},
        "prediction_registered_before_running":
            "passes comfortably -- two parameters over n~400 should barely overfit",
        "n_splits": N_SPLITS, "split": "50/50 random over items, fit on A, score on B",
        "cross_cell_median_heldout_frac": med,
        "cellwise_heldout_range": [float(fr.min()), float(fr.max())],
        "cells_keeping_sign": [n_sign_ok, len(flat)],
        "verdict": "PASS" if passed else "FAIL",
        "cells": cells}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
