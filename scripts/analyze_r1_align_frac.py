"""Align-LoRA P0 — frac genérico-FT multi-ancla (€0 local, post-pod).

Computa, por ancla MC, qué fracción del lift del adapter contrastivo (v7mc) reproduce
el control SFT genérico matched (Align-LoRA r256):

    frac_task = (align_acc - base_acc) / (v7mc_acc - base_acc)

Cierra el hueco del ledger B6b: el ~62% genérico-FT estaba medido SOLO en ARC
(rank-sweep). El pre-registro R1 (2026-06-22) pide el control align también sobre
hellaswag + winogrande. Este script lee los 3 runs R1 (base / v7mc / align) que
produce `run_pod_r5_align_p0.sh` Stage A+B y reporta el frac por ancla.

Lee tanto lm-eval --output_path DIRS (glob results_*.json) como el v7-runner --out
JSON (mismo schema top-level) — misma lógica que check_canonical_repro.py.

Métrica por ancla (la del headline / ledger):
    arc_challenge -> acc_norm   hellaswag -> acc_norm
    winogrande    -> acc        truthfulqa_mc1 -> acc

Usage (local WSL, tras bajar runs/eval_bulletproof/ del pod):
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

# Ancla -> métrica canónica del headline (ledger B9).
TASK_METRIC = {
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "truthfulqa_mc1": "acc",
}


def load_results(path: str) -> dict:
    """Acepta un dir lm-eval (glob results_*.json) o un JSON plano del v7-runner."""
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
    """Devuelve el valor de la métrica o None (tolerante: reportamos lo que haya)."""
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
    ap.add_argument("--out", default=None, help="CSV opcional")
    args = ap.parse_args()

    base = load_results(args.base_dir)
    v7mc = load_results(args.v7mc_json)
    align = load_results(args.align_dir)

    rows = []
    print(f"\n{'ancla':<16} {'metric':<9} {'base':>7} {'v7mc':>7} {'align':>7} "
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
    print("LECTURA: frac alto (≈1) en un ancla => ahí el +lift es ~genérico-FT (NO contraste-especial).")
    print("         frac bajo => ahí el residual contrastivo es real. ARC ya da ~0.62 (ledger B6b).")
    print("         El gate de mecanismo se lee POR ANCLA, no con un solo número.\n")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
