"""E7b -- paired analysis of the missing cell: NEGATING the anti-axis (-theta_anti).

E7 ran three of the four sign x axis combinations in scoring:

    theta_mc  +1   (the canonical adapter)      -> acc_norm .850
    theta_mc  -x   (negating the adapter)       -> converges to chance
    theta_anti +1  (training against)           -> acc_norm .190
    theta_anti -1  (negating the inverse)       -> MISSING  <- this run

The generation face already says -theta_anti is not theta_mc (442 tokens vs 90).
This closes the 2x2 on the scoring face, where the paper's headline lives.

The pre-registered discriminator is quantitative, not qualitative. Geometry gives
cos(theta_anti, -theta_mc) = -0.129, hence cos(theta_anti, theta_mc) = +0.129 and
cos(-theta_anti, theta_mc) = -0.129 -- negating EITHER argument flips the cosine
exactly once, so cos(-a,b) == cos(a,-b). If the lift were a function of alignment
with theta_mc, the cell at scale s should buy about s * 0.129 * lift(theta_mc+1),
i.e. -4.7pp at s=-1 and +4.7pp at s=+1.

That signed, per-scale form is the whole point: a projection model is ANTISYMMETRIC
in the scale, so testing one cell against one number cannot refute it -- a single
cell can agree by coincidence. The refutation lives in the pair.

  history: the first version of this block negated the cosine a second time and
  predicted +4.68pp at s=-1, then read the measured -3.41pp as "wrong sign". With
  the sign right the prediction is -4.68pp, which sits INSIDE that cell's CI. The
  cell alone excludes nothing; what excludes the projection model is s=+1
  (predicted +4.7, measured -28.1) plus the fact that both signs lose accuracy,
  which no antisymmetric model can do.

Everything here is CPU and runs LOCAL (repo rule: the pod is GPU-only). Cells are
paired by doc_hash, so the CI is the one the shared-item design earns -- the
independent-binomial version is conservative by construction (see the L1/C-5
lesson in scripts/l1_paired_bootstrap_recovery_fraction.py).

  python scripts/analyze_e7b_anti_negation.py \
      --cell base=runs/e7_inverse/arc_canonical_scale_0.json \
      --cell theta_mc+1=runs/e7_inverse/arc_canonical_scale_1.json \
      --cell anti+1=runs/e7_inverse/arc_anti_scale_1.json \
      --cell anti-1=runs/e7b_anti_negation/arc_anti_scale_-1.json \
      --out runs/e7b_anti_negation/analysis.json
"""
import argparse
import json
import sys

import numpy as np
from scipy.stats import chisquare

# Verified against artifacts, not memory: runs/e1_plain_ce/eval_base.json and
# runs/e7_inverse/arc_canonical_scale_0.json both read acc_norm 0.5025 on the
# first 400 ARC items under --apply-chat-template --max-length 4096.
# A guard whose constant is written from memory inverts its job (E7 gotcha).
CANONICAL_BASE_ACC_NORM_400 = 0.5025
GUARD_TOL = 0.03
GUARD_SUBSET = 400


def load_cell(path, task="arc_challenge"):
    """Per-item logprobs + continuation lengths, keyed by doc_hash."""
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    samples = blob.get("samples", {}).get(task)
    if not samples:
        sys.exit(f"ABORT: {path} has no embedded samples for {task} "
                 "(was it run with --no-log-samples?)")
    items = {}
    for s in samples:
        lp = np.array([r[0] for r in s["filtered_resps"]], dtype=np.float64)
        nchar = np.array([max(len(a[1]), 1) for a in s["arguments"]], dtype=np.float64)
        items[s["doc_hash"]] = {
            "doc_id": s["doc_id"],
            "logprob": lp,
            "logprob_norm": lp / nchar,
            "target": int(s["target"]),
            # lm-eval's own verdict. Recomputing argmax disagrees on ties (argmax
            # takes the first), which drifts the aggregate by an item or two -- and
            # an aggregate that does not reconcile with its artifact is a finding
            # about the analysis, not about the model.
            "acc": float(s["acc"]),
            "acc_norm": float(s["acc_norm"]),
        }
    published = {}
    for metric, value in blob.get("results", {}).get(task, {}).items():
        if metric.startswith(("acc,", "acc_norm,")):
            published[metric.split(",")[0]] = value
    return items, published


def per_item(cell, channel):
    """Correctness (0/1) and rank of the correct option, item-aligned.

    `hit` is lm-eval's own per-item metric, so the mean reconciles exactly with the
    published aggregate. `rank` is derived from the logprobs but anchored to that
    verdict: a hit is rank 1 by definition, and only ranks 2-4 are read off the
    scores. Free-running argmax disagrees on ties.
    """
    key = "logprob" if channel == "acc" else "logprob_norm"
    hit, rank = {}, {}
    for h, it in cell.items():
        hit[h] = it[channel]
        scores = it[key]
        if it[channel] == 1.0:
            rank[h] = 1
        else:
            rank[h] = max(2, int((scores > scores[it["target"]]).sum()) + 1)
    return hit, rank


def paired_delta(hit_a, hit_b, order, n_boot, seed):
    """Bootstrap over ITEMS, carrying both arms together (a - b), in pp."""
    a = np.array([hit_a[h] for h in order])
    b = np.array([hit_b[h] for h in order])
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    reps = 100.0 * d[idx].mean(axis=1)
    return {
        "delta_pp": float(100.0 * d.mean()),
        "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
        "p_gt_0": float((reps > 0).mean()),
        "se_pp": float(reps.std(ddof=1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable; one lm-eval json per cell")
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--lift-label", default="theta_mc+1",
                    help="cell that defines the full lift, for the projection prediction")
    ap.add_argument("--cos-anti-neg-mc", type=float, default=-0.129,
                    help="cos(theta_anti, -theta_mc) from anti_axis_analysis.json")
    ap.add_argument("--projection-cell", action="append", default=None,
                    metavar="LABEL=SCALE",
                    help="repeatable; cell and the offset scale it was run at. The "
                         "projection prediction is signed by that scale, so passing "
                         "both signs is what makes the test refutable. "
                         "Default: anti-1=-1 and anti+1=1 when present.")
    ap.add_argument("--channel", default="acc_norm", choices=["acc", "acc_norm"])
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-guard", action="store_true",
                    help="only for cells deliberately run off the canonical condition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells, published = {}, {}
    for spec in args.cell:
        if "=" not in spec:
            sys.exit(f"ABORT: --cell wants LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        cells[label], published[label] = load_cell(path, args.task)
        print(f"loaded {label:<14s} n={len(cells[label]):>5d}  {path}")

    if args.base_label not in cells:
        sys.exit(f"ABORT: no cell labelled {args.base_label!r} -- nothing to compare against")

    # --- guard 1: the cells must be the same items, or pairing is a lie ---------
    shared = set.intersection(*(set(c) for c in cells.values()))
    for label, c in cells.items():
        if len(c) != len(shared):
            print(f"!! WARN {label}: {len(c)} items, {len(shared)} shared -> "
                  "restricting every comparison to the shared set")
    order = sorted(shared, key=lambda h: cells[args.base_label][h]["doc_id"])
    if not order:
        sys.exit("ABORT: cells share no items (different --limit or different task?)")

    # --- guard 2: the eval condition must be the canonical one -----------------
    base_hit_norm, _ = per_item(cells[args.base_label], "acc_norm")
    first = sorted(order, key=lambda h: cells[args.base_label][h]["doc_id"])[:GUARD_SUBSET]
    base_400 = float(np.mean([base_hit_norm[h] for h in first]))
    guard_ok = abs(base_400 - CANONICAL_BASE_ACC_NORM_400) <= GUARD_TOL
    print(f"\nrepro guard: base acc_norm on the first {len(first)} items = {base_400:.4f} "
          f"(canonical {CANONICAL_BASE_ACC_NORM_400}) -> {'OK' if guard_ok else 'FAIL'}")
    if not guard_ok and not args.skip_guard:
        sys.exit("ABORT: the eval condition drifted (chat template / max_length). No cell\n"
                 "       here is comparable to the +35.83 headline. This is the C-3 failure mode.")

    # --- the table -------------------------------------------------------------
    hits = {lbl: per_item(c, args.channel)[0] for lbl, c in cells.items()}
    ranks = {lbl: per_item(c, "acc")[1] for lbl, c in cells.items()}
    out = {"task": args.task, "channel": args.channel, "n_paired": len(order),
           "base_acc_norm_first400": base_400, "cells": {}}

    print(f"\n=== {args.task} / {args.channel} / n={len(order)} paired ===")
    print(f"{'cell':<14s} {'acc':>7s} {'published':>10s} {'delta pp':>9s} "
          f"{'95% CI (paired)':>20s} {'P(d>0)':>7s}")
    for label in cells:
        acc = float(np.mean([hits[label][h] for h in order]))
        row = {"acc": acc, "published": published[label].get(args.channel)}
        if label != args.base_label:
            row.update(paired_delta(hits[label], hits[args.base_label], order,
                                    args.n_boot, args.seed))
            ci = f"[{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]"
            print(f"{label:<14s} {acc:>7.4f} {str(row['published']):>10s} "
                  f"{row['delta_pp']:>+9.2f} {ci:>20s} {row['p_gt_0']:>7.3f}")
        else:
            print(f"{label:<14s} {acc:>7.4f} {str(row['published']):>10s} "
                  f"{'--':>9s} {'--':>20s} {'--':>7s}")
        out["cells"][label] = row

    # --- rank distribution: concentrate / disperse / displace ------------------
    print(f"\n=== rank of the correct option (channel acc, n={len(order)}) ===")
    print(f"{'cell':<14s} {'#1':>6s} {'#2':>6s} {'#3':>6s} {'#4':>6s} "
          f"{'chi2 vs uniform':>16s} {'p':>9s}")
    for label in cells:
        counts = np.bincount([ranks[label][h] for h in order], minlength=5)[1:5]
        chi2, p = chisquare(counts)
        pct = 100.0 * counts / counts.sum()
        print(f"{label:<14s} {pct[0]:>6.1f} {pct[1]:>6.1f} {pct[2]:>6.1f} {pct[3]:>6.1f} "
              f"{chi2:>16.2f} {p:>9.2g}")
        out["cells"][label]["rank_pct"] = pct.tolist()
        out["cells"][label]["rank_chi2"] = float(chi2)
        out["cells"][label]["rank_chi2_p"] = float(p)

    # --- the pre-registered discriminator --------------------------------------
    # cos(theta_anti, theta_mc) = -cos(theta_anti, -theta_mc). The cell run at offset
    # scale s injects s*theta_anti, whose projection on theta_mc carries the sign of s.
    cos_anti_mc = -args.cos_anti_neg_mc
    specs = args.projection_cell
    if specs is None:
        specs = [f"{lbl}={s}" for lbl, s in (("anti-1", -1.0), ("anti+1", 1.0))
                 if lbl in out["cells"]]
    if args.lift_label in out["cells"] and specs:
        full_lift = out["cells"][args.lift_label]["delta_pp"]
        print(f"\n=== projection prediction (lift ~ alignment with theta_mc) ===")
        print(f"  cos(theta_anti, theta_mc) = {cos_anti_mc:+.3f}   "
              f"full lift ({args.lift_label}) = {full_lift:+.2f} pp")
        print(f"  {'cell':<10s} {'scale':>6s} {'predicted':>10s} {'observed':>10s} "
              f"{'95% CI (paired)':>20s}  verdict")
        tests, excluded_any, signs = {}, False, set()
        for spec in specs:
            if "=" not in spec:
                sys.exit(f"ABORT: --projection-cell wants LABEL=SCALE, got {spec!r}")
            label, raw = spec.split("=", 1)
            if label not in out["cells"]:
                sys.exit(f"ABORT: --projection-cell names {label!r}, which is not a loaded cell")
            scale = float(raw)
            predicted = scale * cos_anti_mc * full_lift
            got = out["cells"][label]
            lo, hi = got["ci95"]
            excluded = not (lo <= predicted <= hi)
            excluded_any |= excluded
            signs.add(np.sign(got["delta_pp"]))
            print(f"  {label:<10s} {scale:>+6.2f} {predicted:>+10.2f} "
                  f"{got['delta_pp']:>+10.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>20s}  "
                  f"{'EXCLUDED' if excluded else 'not excluded'}"
                  f"{'' if (lo <= 0 <= hi) else ' (zero excluded)'}")
            tests[label] = {"scale": scale, "predicted_pp": predicted,
                            "observed_pp": got["delta_pp"], "observed_ci95": got["ci95"],
                            "linear_excluded": excluded,
                            "zero_excluded": not (lo <= 0 <= hi)}
        # A projection model is antisymmetric in the scale: opposite scales must give
        # opposite-signed deltas. Same sign on both is a refutation no single cell gives.
        one_sided = len(signs) == 1 and len([s for s in specs]) > 1
        print(f"  -> LINEAR-IN-PROJECTION {'EXCLUDED' if (excluded_any or one_sided) else 'not excluded'}"
              f"{'; one-sided: every scale moves the same way, which no antisymmetric model can do' if one_sided else ''}")
        out["projection_test"] = {"cos_anti_vs_mc": cos_anti_mc,
                                  "cos_neg_anti_vs_mc": args.cos_anti_neg_mc,
                                  "full_lift_pp": full_lift,
                                  "cells": tests,
                                  "linear_excluded_any_cell": excluded_any,
                                  "one_sided_response": bool(one_sided),
                                  "linear_excluded": bool(excluded_any or one_sided)}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
