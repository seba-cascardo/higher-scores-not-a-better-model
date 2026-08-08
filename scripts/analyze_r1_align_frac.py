"""Align-LoRA P0 — generic-FT fraction across MC anchors (local, CPU-only).

Per MC anchor, what fraction of the contrastive adapter's lift (v7mc) does the
matched generic SFT control (Align-LoRA r256) reproduce:

    frac_task = (align_acc - base_acc) / (v7mc_acc - base_acc)

The ~62% generic-FT figure was first measured on ARC alone (rank sweep). The R1
pre-registration (2026-06-22) asks for the align control on hellaswag and
winogrande as well. This script reads the three R1 runs (base / v7mc / align)
and reports the fraction per anchor.

Reads both lm-eval --output_path DIRS (glob results_*.json) and the v7-runner --out
JSON (same top-level schema) — same logic as check_canonical_repro.py.

Metric per anchor (the headline metric):
    arc_challenge -> acc_norm   hellaswag -> acc_norm
    winogrande    -> acc        truthfulqa_mc1 -> acc

Usage (after fetching runs/eval_bulletproof/ from the evaluation host):
    python scripts/analyze_r1_align_frac.py \
        --base-dir  runs/eval_bulletproof/R1_base \
        --v7mc-json runs/eval_bulletproof/R1_v7mc.json \
        --align-dir runs/eval_bulletproof/R1_align \
        --out       runs/eval_bulletproof/r1_align_frac.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# The report prints non-ASCII (approx sign); a Windows console defaults to
# cp1252 and would die mid-report instead of finishing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Anchor -> the canonical headline metric.
TASK_METRIC = {
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "truthfulqa_mc1": "acc",
}


def load_results(path: str) -> dict:
    """Accept an lm-eval dir (glob results_*.json) or a flat v7-runner JSON."""
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("**/results_*.json"))
        if not cands:
            sys.exit(f"FAIL: no results_*.json under {path}")
        p = cands[-1]
    if not p.exists():
        sys.exit(f"FAIL: results not found at {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_metric(results: dict, task: str, metric: str):
    """Return the metric value, or None (tolerant: we report whatever is there)."""
    res = results.get("results", {})
    if task not in res:
        return None
    r = res[task]
    for k in (f"{metric},none", metric, f"{metric},strict-match", f"{metric},flexible-extract"):
        if k in r:
            try:
                return float(r[k])
            except (TypeError, ValueError):
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="R1_base (lm-eval dir)")
    ap.add_argument("--v7mc-json", required=True, help="R1_v7mc.json (v7-runner --out)")
    ap.add_argument("--align-dir", required=True, help="R1_align (lm-eval dir)")
    ap.add_argument("--out", default=None, help="optional CSV")
    args = ap.parse_args()

    base = load_results(args.base_dir)
    v7mc = load_results(args.v7mc_json)
    align = load_results(args.align_dir)

    rows = []
    print(f"\n{'anchor':<16} {'metric':<9} {'base':>7} {'v7mc':>7} {'align':>7} "
          f"{'lift_v7':>8} {'lift_al':>8} {'frac':>7}")
    print("-" * 80)
    for task, metric in TASK_METRIC.items():
        b = get_metric(base, task, metric)
        v = get_metric(v7mc, task, metric)
        a = get_metric(align, task, metric)
        if None in (b, v, a):
            print(f"{task:<16} {metric:<9}  [missing: base={b} v7mc={v} align={a}]")
            rows.append({"task": task, "metric": metric, "base": b, "v7mc": v,
                         "align": a, "lift_v7mc": None, "lift_align": None, "frac": None})
            continue
        lift_v = v - b
        lift_a = a - b
        frac = (lift_a / lift_v) if abs(lift_v) > 1e-9 else float("nan")
        print(f"{task:<16} {metric:<9} {b:>7.4f} {v:>7.4f} {a:>7.4f} "
              f"{lift_v:>+8.4f} {lift_a:>+8.4f} {frac:>7.3f}")
        rows.append({"task": task, "metric": metric, "base": round(b, 4),
                     "v7mc": round(v, 4), "align": round(a, 4),
                     "lift_v7mc": round(lift_v, 4), "lift_align": round(lift_a, 4),
                     "frac": round(frac, 4)})

    print("-" * 80)
    print("READ: frac near 1 on an anchor => there the lift is ~generic-FT (not contrast-specific).")
    print("      frac low => there the contrastive residual is real. ARC already gives ~0.62.")
    print("      The mechanism gate is read PER ANCHOR, not from a single number.\n")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
