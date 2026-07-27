"""R1 (audit remediation) -- does the affine margin map PREDICT out of sample, and does
it beat the trivial alternatives?

The published §5.9 claim rests on fitting ga ~ a*gb + c per cell and reporting how much
of the accuracy change two scalars reproduce. That fit is IN-SAMPLE
(`analyze_threshold_density_mechanism.py:167`). A first version of this script validated
it out of sample but with a criterion derived from the very in-sample range it was
meant to validate ([0.70, 1.93]) -- circular, and a median of 1.8 would have "passed".
It also read a RATIO whose denominator is near zero in the small-effect cells. Both
defects are fixed here.

What this script measures, per cell, under 5-fold cross-fitting (every item is scored
by parameters estimated without it):

  gap = logp(gold) - logp(best distractor), as in the canonical script
  fit (a, c) on the training folds -> predict on the held-out fold ->
  d_acc_pred = mean[(a*gb + c) > 0] - mean[gb > 0]      (pooled over folds)
  ABSOLUTE ERROR in percentage points against the measured d_acc  <-- primary metric

and compares the affine map against four nested/trivial baselines, all cross-fitted the
same way:

  identity      ga = gb                 (predicts no accuracy change at all)
  intercept     ga = gb + c             (a pure shift: no rescaling)
  slope         ga = a*gb               (a pure rescaling: no shift)
  constant      ga = c                  (ignores the base margin entirely)
  affine        ga = a*gb + c

The intercept-only baseline is the one that matters. The paper claims the accuracy
change needs BOTH scalars -- "scale is one scalar; the accuracy change needs a second".
If a pure shift predicts as well as the affine map, that claim is not supported and
§5.9 must be restated as a threshold shift rather than a rescaling.

CRITERION FIXED BEFORE RUNNING (plans/2026-07-27-external-audit-remediation.md):
  A cell is NON-NULL if |measured d_acc| >= 3.0 pp (14 of the 18 cells qualify; the
  four excluded measure +2.17, +0.83, +2.33 and exactly 0.00 pp).
  PASS requires ALL THREE:
    (1) median absolute error of the affine map over all cells <= 3.0 pp
    (2) >= 12 of the 14 non-null cells predicted with the correct sign
    (3) the affine map's median absolute error is strictly lower than BOTH
        intercept-only and slope-only
  Failing (3) alone does not kill the section -- it demotes "rescales the margin" to
  whichever single-parameter description survives, and the wording follows the data.

This script decides predictive adequacy of a descriptive signature ONLY. It does not
establish item-independence, label-freeness, or a causal mechanism: the fit lives in
gold-relative margin coordinates, which are defined using the label.

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
N_FOLDS = 5
N_REPEATS = 40          # repeated 5-fold; every repeat reshuffles the fold assignment
SEED = 0
NONNULL_PP = 3.0        # |measured d_acc| threshold for a cell to count toward the sign rule
MAX_MEDIAN_ABS_ERR_PP = 3.0
MIN_SIGN_OK = 12

MODELS = ("affine", "intercept", "slope", "constant", "identity")

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
    """(gb, ga) on shared items -- same logic as the canonical script."""
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


def fit_predict(name, gb_tr, ga_tr, gb_te):
    """Estimate a model's parameters on the training folds, apply to the held-out fold."""
    if name == "affine":
        a, c = np.polyfit(gb_tr, ga_tr, 1)
    elif name == "intercept":                       # pure shift, no rescaling
        a, c = 1.0, float((ga_tr - gb_tr).mean())
    elif name == "slope":                           # pure rescaling, no shift
        denom = float((gb_tr * gb_tr).sum())
        a = float((gb_tr * ga_tr).sum() / denom) if denom else 1.0
        c = 0.0
    elif name == "constant":                        # ignores the base margin
        a, c = 0.0, float(ga_tr.mean())
    elif name == "identity":
        a, c = 1.0, 0.0
    else:
        raise ValueError(name)
    return a, c, a * gb_te + c


def crossfit(gb, ga, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=SEED):
    """Repeated k-fold cross-fitting. Every item is predicted by params fit without it."""
    rng = np.random.default_rng(seed)
    n = len(gb)
    acc_b_all = float((gb > 0).mean())
    acc_a_all = float((ga > 0).mean())
    d_measured = (acc_a_all - acc_b_all) * 100

    per_model = {m: {"d_pred": [], "a": [], "c": []} for m in MODELS}
    r2 = []
    for _ in range(n_repeats):
        idx = rng.permutation(n)
        folds = np.array_split(idx, n_folds)
        pred = {m: np.empty(n) for m in MODELS}
        r2_num, r2_den = 0.0, 0.0
        for f in folds:
            tr = np.setdiff1d(idx, f, assume_unique=False)
            if len(tr) < 10 or len(f) < 2:
                continue
            for m in MODELS:
                a, c, p = fit_predict(m, gb[tr], ga[tr], gb[f])
                pred[m][f] = p
                if m == "affine":
                    per_model[m]["a"].append(a)
                    per_model[m]["c"].append(c)
            r2_num += float(((ga[f] - pred["affine"][f]) ** 2).sum())
            r2_den += float(((ga[f] - ga[tr].mean()) ** 2).sum())
        for m in MODELS:
            per_model[m]["d_pred"].append((float((pred[m] > 0).mean()) - acc_b_all) * 100)
        if r2_den:
            r2.append(1.0 - r2_num / r2_den)

    out = {"n_items": n, "acc_base": acc_b_all, "acc_arm": acc_a_all,
           "d_acc_measured_pp": d_measured,
           "is_non_null": bool(abs(d_measured) >= NONNULL_PP),
           "r2_heldout_affine": float(np.mean(r2)) if r2 else None,
           "affine_slope_mean": float(np.mean(per_model["affine"]["a"])),
           "affine_slope_sd_across_folds": float(np.std(per_model["affine"]["a"], ddof=1)),
           "affine_intercept_mean": float(np.mean(per_model["affine"]["c"])),
           "affine_intercept_sd_across_folds": float(np.std(per_model["affine"]["c"], ddof=1)),
           "models": {}}
    for m in MODELS:
        dp = float(np.mean(per_model[m]["d_pred"]))
        out["models"][m] = {
            "d_acc_pred_pp": dp,
            "abs_err_pp": abs(dp - d_measured),
            "sign_ok": (bool(np.sign(dp) == np.sign(d_measured))
                        if d_measured != 0 else None),
        }
    return out


def main():
    print("=" * 108)
    print("R1 -- cross-fitted validation of the affine margin map, against trivial baselines")
    print(f"     {N_REPEATS}x {N_FOLDS}-fold cross-fitting; primary metric = |predicted - measured| d_acc, in pp")
    print("=" * 108)
    print(f"  {'arm':<28}{'task':<16}{'meas':>8}" +
          "".join(f"{m[:6]:>9}" for m in MODELS) + f"{'R2':>7}")

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
                if len(gb):
                    seeds.append(crossfit(gb, ga))
                print(f"    [{done}/{total}] {label} | {t} | {os.path.basename(p)}", flush=True)
            if not seeds:
                continue

            agg = {"n_seeds": len(seeds), "n_items": seeds[0]["n_items"],
                   "d_acc_measured_pp": float(np.mean([s["d_acc_measured_pp"] for s in seeds])),
                   "r2_heldout_affine": float(np.mean([s["r2_heldout_affine"] for s in seeds
                                                       if s["r2_heldout_affine"] is not None])),
                   "affine_slope_mean": float(np.mean([s["affine_slope_mean"] for s in seeds])),
                   "affine_slope_sd_across_folds":
                       float(np.mean([s["affine_slope_sd_across_folds"] for s in seeds])),
                   "affine_intercept_mean": float(np.mean([s["affine_intercept_mean"] for s in seeds])),
                   "affine_intercept_sd_across_folds":
                       float(np.mean([s["affine_intercept_sd_across_folds"] for s in seeds])),
                   "models": {}}
            agg["is_non_null"] = bool(abs(agg["d_acc_measured_pp"]) >= NONNULL_PP)
            for m in MODELS:
                agg["models"][m] = {
                    "d_acc_pred_pp": float(np.mean([s["models"][m]["d_acc_pred_pp"] for s in seeds])),
                    "abs_err_pp": float(np.mean([s["models"][m]["abs_err_pp"] for s in seeds])),
                }
                dp, dm = agg["models"][m]["d_acc_pred_pp"], agg["d_acc_measured_pp"]
                agg["models"][m]["sign_ok"] = (bool(np.sign(dp) == np.sign(dm)) if dm else None)
            per_task[t] = agg
            print(f"  {label:<28}{t:<16}{agg['d_acc_measured_pp']:>+8.2f}" +
                  "".join(f"{agg['models'][m]['abs_err_pp']:>9.2f}" for m in MODELS) +
                  f"{agg['r2_heldout_affine']:>7.3f}", flush=True)
            flat.append((label, t, agg))
        cells[label] = {"family": fam, "objective": obj, "per_task": per_task}

    # --- verdict ---------------------------------------------------------------
    med = {m: float(np.median([c[2]["models"][m]["abs_err_pp"] for c in flat])) for m in MODELS}
    nonnull = [c for c in flat if c[2]["is_non_null"]]
    sign_ok = sum(1 for c in nonnull if c[2]["models"]["affine"]["sign_ok"])
    within3 = sum(1 for c in flat if c[2]["models"]["affine"]["abs_err_pp"] <= 3.0)

    c1 = med["affine"] <= MAX_MEDIAN_ABS_ERR_PP
    c2 = sign_ok >= MIN_SIGN_OK
    c3 = med["affine"] < med["intercept"] and med["affine"] < med["slope"]
    passed = bool(c1 and c2 and c3)

    print("\n" + "=" * 108)
    print("VERDICT (criterion fixed before running)")
    print("=" * 108)
    print("  median |predicted - measured| d_acc, in pp, cross-fitted:")
    for m in MODELS:
        print(f"      {m:<12}{med[m]:>7.2f} pp")
    print(f"\n  (1) affine median abs err <= {MAX_MEDIAN_ABS_ERR_PP} pp        : "
          f"{med['affine']:.2f}  -> {'PASS' if c1 else 'FAIL'}")
    print(f"  (2) sign correct in >= {MIN_SIGN_OK} of {len(nonnull)} non-null cells: "
          f"{sign_ok}  -> {'PASS' if c2 else 'FAIL'}")
    print(f"  (3) affine beats intercept-only AND slope-only : "
          f"{med['affine']:.2f} vs {med['intercept']:.2f} / {med['slope']:.2f}"
          f"  -> {'PASS' if c3 else 'FAIL'}")
    print(f"\n  cells within +-3pp: {within3}/{len(flat)}")
    print(f"\n  >>> {'PASS' if passed else 'FAIL'} <<<")
    if not c3:
        print("  NOTE: criterion (3) failed -- a single-parameter description predicts as")
        print("  well as the affine map. §5.9 must be restated accordingly.")

    out = {"criterion_fixed_before_running": {
        "primary_metric": "|predicted - measured| accuracy change, in pp, cross-fitted",
        "non_null_cell": f"|measured d_acc| >= {NONNULL_PP} pp",
        "1_median_abs_err_pp": f"<= {MAX_MEDIAN_ABS_ERR_PP}",
        "2_sign_correct_non_null": f">= {MIN_SIGN_OK}",
        "3_beats_baselines": "affine median abs err < intercept-only AND < slope-only",
        "scope": ("decides predictive adequacy of a descriptive signature only; not "
                  "item-independence, not label-freeness, not causality -- the fit is "
                  "in gold-relative margin coordinates, which use the label")},
        "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
        "median_abs_err_pp": med,
        "non_null_cells": len(nonnull), "sign_ok_non_null": sign_ok,
        "cells_within_3pp": [within3, len(flat)],
        "criteria_met": {"median_abs_err": bool(c1), "sign": bool(c2),
                         "beats_baselines": bool(c3)},
        "verdict": "PASS" if passed else "FAIL",
        "cells": cells}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
