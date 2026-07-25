"""L1 — PAIRED bootstrap (by doc_id) of the generic-recovery fraction.

Supersedes the C-5 check (`validate_frac_ci_bootstrap.py`), which resampled each
arm as an INDEPENDENT binomial from published aggregate accuracies. That is
conservative by construction: the three arms are evaluated on the SAME items, so
pairing induces positive correlation between arms, which shrinks the variance of
both the numerator (arm - base) and the denominator (v7mc - base). C-5 could
therefore only show the published delta-method interval was not too narrow; it
could not deliver the interval the paired design actually earns.

This script resamples ITEMS (doc_id) with replacement, carrying all three arms
together, and recomputes

    frac = (align - base) / (v7mc - base)

per replicate. Reports the percentile CI, P(frac > 0.5), and the induced
arm-to-arm correlation that explains any shrinkage vs C-5.

Data (all local, per-item, n=400 per task):
  base   runs/eval_bulletproof/R1_base/**/samples_<task>_2026-06-29*.jsonl
  align  runs/eval_bulletproof/R1_align/runs__align_lora_control__r256/samples_<task>_*.jsonl
  v7mc   runs/eval_bulletproof/R1_v7mc.json  (samples embedded)
  null   runs/eval_bulletproof/R1_null.json  (zeroed-offset control, sanity only)

A hard guard aborts before the expensive stage if any arm's recomputed aggregate
disagrees with the published value, or if the three arms are not item-aligned by
doc_hash -- the pattern that has already saved two pod runs.

  python scripts/l1_paired_bootstrap_recovery_fraction.py
"""
import glob
import json
import sys

import numpy as np

B = 200_000          # match C-5's replicate count so the intervals are comparable
SEED = 0
OUT = "runs/oq1_functional_axis/l1_paired_bootstrap.json"

# --metric acc reruns the identical paired bootstrap on plain accuracy. ARC and
# HellaSwag are published under acc_norm, so this is the robustness arm asking how
# much of the recovery fraction is length normalisation (FASE 1 found HellaSwag's
# inducibility collapses from 69.5% to 8.9% under acc in the same-param control).
# Winogrande and TruthfulQA define no acc_norm, so they come out identical.
# The canonical path is untouched: this writes a different file and skips the
# published-value guard, whose reference values are acc_norm by construction.
METRIC_OVERRIDE = None
OUT_ACC = "runs/oq1_functional_axis/l1_paired_bootstrap_acc.json"

# task -> (metric, published base, published v7mc, published align_r256)
# published values from runs/eval_bulletproof/r1_align_frac.csv (canonical R1 arms)
TASKS = {
    "arc_challenge":  ("acc_norm", 0.505,  0.855,  0.7225),
    "hellaswag":      ("acc_norm", 0.540,  0.710,  0.6975),
    "winogrande":     ("acc",      0.6575, 0.8475, 0.7750),
    "truthfulqa_mc1": ("acc",      0.4375, 0.810,  0.7775),
}
TOL = 5e-4

BASE_GLOB = "runs/eval_bulletproof/R1_base/**/samples_{task}_2026-06-29*.jsonl"
ALIGN_GLOB = ("runs/eval_bulletproof/R1_align/runs__align_lora_control__r256/"
              "samples_{task}_*.jsonl")
V7MC_JSON = "runs/eval_bulletproof/R1_v7mc.json"
NULL_JSON = "runs/eval_bulletproof/R1_null.json"


def load_jsonl(pattern, task):
    paths = sorted(glob.glob(pattern.format(task=task), recursive=True))
    if len(paths) != 1:
        sys.exit(f"[ABORT] expected exactly 1 file for {task}, got {len(paths)}: {paths}")
    with open(paths[0], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows, paths[0]


def load_embedded(path, task):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["samples"][task]


def vec(rows, metric):
    """Per-item metric in doc_id order, plus the doc_hash order key."""
    rows = sorted(rows, key=lambda r: r["doc_id"])
    y = np.array([float(r[metric]) for r in rows])
    h = [r["doc_hash"] for r in rows]
    return y, h


def main():
    rng = np.random.default_rng(SEED)
    out = {"bootstrap_reps": B, "seed": SEED, "resampling": "paired by doc_id (items resampled together across arms)",
           "supersedes": "runs/oq1_functional_axis/frac_ci_bootstrap.json (C-5, independent binomial per arm)",
           "per_task": []}

    for task, (metric, p_base, p_v7, p_align) in TASKS.items():
        if METRIC_OVERRIDE:
            metric = METRIC_OVERRIDE
        print(f"[{task}] loading arms...  (metric={metric})", flush=True)
        base_rows, base_path = load_jsonl(BASE_GLOB, task)
        align_rows, align_path = load_jsonl(ALIGN_GLOB, task)
        v7_rows = load_embedded(V7MC_JSON, task)
        null_rows = load_embedded(NULL_JSON, task)

        y_base, h_base = vec(base_rows, metric)
        y_align, h_align = vec(align_rows, metric)
        y_v7, h_v7 = vec(v7_rows, metric)
        y_null, _ = vec(null_rows, metric)

        # --- GUARD 1: item alignment (same items, same order, by content hash) ---
        n = len(y_base)
        if not (len(y_align) == len(y_v7) == n):
            sys.exit(f"[ABORT] {task}: arm lengths differ ({n}/{len(y_align)}/{len(y_v7)})")
        if not (h_base == h_align == h_v7):
            bad = sum(1 for a, b, c in zip(h_base, h_align, h_v7) if not (a == b == c))
            sys.exit(f"[ABORT] {task}: {bad}/{n} items not aligned across arms by doc_hash")

        # --- GUARD 2: aggregates reproduce the published numbers ---
        # Under --metric acc the published references are acc_norm and would abort
        # by construction -- but the guard is doing PROVENANCE ("did I load the
        # right arms?"), not just metric agreement, so dropping it entirely would
        # lose that. The rows carry both channels, so the check still runs on the
        # published channel even while the bootstrap runs on the other one.
        guard_metric = TASKS[task][0]
        for name, rows, pub in (("base", base_rows, p_base), ("v7mc", v7_rows, p_v7),
                                ("align", align_rows, p_align)):
            y_guard, _ = vec(rows, guard_metric)
            if abs(y_guard.mean() - pub) > TOL:
                sys.exit(f"[ABORT] {task}/{name}: recomputed {y_guard.mean():.4f} != "
                         f"published {pub:.4f} on {guard_metric}")
        print(f"  guards OK  n={n}  base={y_base.mean():.4f} v7mc={y_v7.mean():.4f} "
              f"align={y_align.mean():.4f}  null={y_null.mean():.4f}", flush=True)

        frac_point = (y_align.mean() - y_base.mean()) / (y_v7.mean() - y_base.mean())

        # --- paired bootstrap: resample ITEMS, carry all arms together ---
        # chunked so the index matrix never exceeds ~1e7 entries at once
        chunk = 20_000
        parts, degenerate = [], 0
        for start in range(0, B, chunk):
            k = min(chunk, B - start)
            idx = rng.integers(0, n, size=(k, n))
            mb = y_base[idx].mean(1)
            ma = y_align[idx].mean(1)
            mv = y_v7[idx].mean(1)
            den = mv - mb
            degenerate += int((den <= 0).sum())
            ok = den > 0
            parts.append((ma[ok] - mb[ok]) / den[ok])
            print(f"    [{start + k}/{B}] bootstrap replicates", flush=True)
        fr = np.concatenate(parts)

        lo, hi = np.percentile(fr, [2.5, 97.5])
        p_gt_half = float((fr > 0.5).mean())

        # why it shrinks: item-level correlation between arms
        corr_ab = float(np.corrcoef(y_base, y_align)[0, 1])
        corr_bv = float(np.corrcoef(y_base, y_v7)[0, 1])

        rec = {
            "task": task, "metric": metric, "n": n,
            "base": float(y_base.mean()), "v7mc": float(y_v7.mean()),
            "align_r256": float(y_align.mean()), "null_adapter": float(y_null.mean()),
            "frac_point": float(frac_point),
            "frac_ci95_paired": [float(lo), float(hi)],
            "frac_median": float(np.median(fr)),
            "p_frac_gt_0.5": p_gt_half,
            "item_corr_base_align": corr_ab,
            "item_corr_base_v7mc": corr_bv,
            "degenerate_replicates": degenerate,
            "sources": {"base": base_path, "align": align_path, "v7mc": V7MC_JSON},
        }
        out["per_task"].append(rec)
        print(f"  frac {frac_point:.4f}  paired CI95 [{lo:.3f}, {hi:.3f}]  "
              f"P(f>0.5)={p_gt_half:.3f}  width={hi - lo:.3f}", flush=True)

    if METRIC_OVERRIDE:
        # The published intervals below are acc_norm; comparing them to an acc run
        # would be an apples-to-oranges shrinkage number. Write and stop.
        out["metric_override"] = METRIC_OVERRIDE
        out["note"] = ("robustness arm: identical paired bootstrap on plain accuracy. "
                       "Winogrande/TruthfulQA define no acc_norm and are unchanged by "
                       "construction; ARC and HellaSwag are the informative rows.")
        with open(OUT_ACC, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[done] wrote {OUT_ACC}", flush=True)
        return

    # --- ARC headline comparison against the two published intervals ---
    arc = next(r for r in out["per_task"] if r["task"] == "arc_challenge")
    lo, hi = arc["frac_ci95_paired"]
    pub_lo, pub_hi = 0.45, 0.85
    c5_lo, c5_hi = 0.49677419354838726, 0.798611111111111
    out["arc_comparison"] = {
        "paired": [lo, hi], "paired_width": hi - lo,
        "published_delta_method": [pub_lo, pub_hi], "published_width": pub_hi - pub_lo,
        "c5_independent_binomial": [c5_lo, c5_hi], "c5_width": c5_hi - c5_lo,
        "shrinkage_vs_published": 1 - (hi - lo) / (pub_hi - pub_lo),
        "shrinkage_vs_c5": 1 - (hi - lo) / (c5_hi - c5_lo),
        "contained_in_published": bool(lo >= pub_lo - 1e-9 and hi <= pub_hi + 1e-9),
    }
    print(f"\n[ARC] paired [{lo:.3f}, {hi:.3f}] (w={hi - lo:.3f})  vs  published "
          f"[{pub_lo}, {pub_hi}] (w={pub_hi - pub_lo:.3f})  vs  C-5 [{c5_lo:.3f}, {c5_hi:.3f}]",
          flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    # Parsed by hand rather than with argparse to keep the canonical path a plain
    # `python <script>`. Three failure modes are rejected explicitly, because the
    # cost of any of them is silently overwriting the CANONICAL artifact with
    # numbers from a different metric:
    #   --metric=acc     an `in sys.argv` test does not match it (repo precedent:
    #                    44ed4b0, argparse and a leading '-')
    #   --metric         with nothing after it -> IndexError
    #   --anything-else  ignored without a word
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--metric":
            if i + 1 >= len(argv):
                sys.exit("[ABORT] --metric needs a value: acc")
            METRIC_OVERRIDE = argv[i + 1]
        elif a.startswith("--metric="):
            METRIC_OVERRIDE = a.split("=", 1)[1]
        elif i == 0 or argv[i - 1] != "--metric":
            sys.exit(f"[ABORT] unknown argument {a!r}; the only flag is --metric acc")
    if METRIC_OVERRIDE is not None and METRIC_OVERRIDE != "acc":
        # acc_norm is not accepted even though it looks legal: Winogrande and
        # TruthfulQA rows carry no acc_norm field, so it would die mid-run with a
        # KeyError after burning the ARC and HellaSwag bootstraps. The canonical
        # run already IS acc_norm -- run it with no flag.
        sys.exit(f"[ABORT] --metric only accepts 'acc', got {METRIC_OVERRIDE!r}. "
                 "For acc_norm run the script with no flag (that is the canonical path).")
    main()
