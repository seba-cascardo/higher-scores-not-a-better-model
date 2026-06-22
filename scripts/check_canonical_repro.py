"""Gate checker for the bullet-proof eval batch (pre-reg 2026-06-22).

Three modes, all exit 1 on FAIL so the pod batch can `|| exit 1` (auto-abort)
or `|| echo WARN` as appropriate. Reads both lm-eval --output_path DIRS (globs
results_*.json) and the v7 runner's single --out JSON (same top-level schema).

Usage:
  # reproduce: base ~= expected AND adapter ~= expected (wrong-base/offsets guard)
  python scripts/check_canonical_repro.py --mode reproduce \
    --base-dir runs/eval_bulletproof/R1_base --adapter-json runs/eval_bulletproof/R1_v7mc.json \
    --task arc_challenge --metric acc_norm --expect-base 0.505 --expect-adapter 0.855 --tol 0.05

  # null: the null_adapter run must equal base within tol (wrapper-identity guard)
  python scripts/check_canonical_repro.py --mode null \
    --base-dir runs/eval_bulletproof/R1_base --adapter-json runs/eval_bulletproof/R1_null.json \
    --task arc_challenge --metric acc_norm --tol 0.02

  # control-gate: align does SOMETHING (not a no-op) AND its lift << v7mc lift
  python scripts/check_canonical_repro.py --mode control-gate \
    --base-dir runs/eval_bulletproof/R1_base --align-dir runs/eval_bulletproof/R1_align \
    --adapter-json runs/eval_bulletproof/R1_v7mc.json \
    --task arc_challenge --metric acc_norm --min-effect 0.01 --max-frac 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("**/results_*.json"))
        if not cands:
            sys.exit(f"FAIL: no results_*.json under {path}")
        p = cands[-1]
    if not p.exists():
        sys.exit(f"FAIL: results not found at {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_metric(path: str, task: str, metric: str) -> float:
    res = load_results(path).get("results", {})
    if task not in res:
        sys.exit(f"FAIL: task {task} not in results of {path} (have: {list(res)[:8]})")
    r = res[task]
    for k in (f"{metric},none", metric, f"{metric},strict-match", f"{metric},flexible-extract"):
        if k in r:
            return float(r[k])
    sys.exit(f"FAIL: metric {metric} not found for {task} in {path} (keys: {list(r)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["reproduce", "null", "control-gate"])
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--metric", default="acc_norm")
    ap.add_argument("--base-dir")
    ap.add_argument("--adapter-json")
    ap.add_argument("--align-dir")
    ap.add_argument("--expect-base", type=float)
    ap.add_argument("--expect-adapter", type=float)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-effect", type=float, default=0.01)
    ap.add_argument("--max-frac", type=float, default=0.5)
    args = ap.parse_args()

    t, m = args.task, args.metric

    if args.mode == "reproduce":
        base = get_metric(args.base_dir, t, m)
        adap = get_metric(args.adapter_json, t, m)
        db = abs(base - args.expect_base)
        da = abs(adap - args.expect_adapter)
        print(f"[reproduce] {t}/{m}: base={base:.4f} (exp {args.expect_base}, |Δ|={db:.4f}) "
              f"adapter={adap:.4f} (exp {args.expect_adapter}, |Δ|={da:.4f}) tol={args.tol}")
        if db > args.tol or da > args.tol:
            sys.exit(f"FAIL: canonical NOT reproduced (check BASE path / offsets_mc.pt). "
                     f"base|Δ|={db:.4f}, adapter|Δ|={da:.4f} > tol {args.tol}")
        print("PASS: canonical reproduced.")

    elif args.mode == "null":
        base = get_metric(args.base_dir, t, m)
        null = get_metric(args.adapter_json, t, m)
        d = abs(base - null)
        print(f"[null] {t}/{m}: base={base:.4f} null_adapter={null:.4f} |Δ|={d:.4f} tol={args.tol}")
        if d > args.tol:
            sys.exit(f"FAIL: null_adapter != base (|Δ|={d:.4f} > {args.tol}). The v7hf wrapper "
                     f"codepath is NOT identity at alpha=theta=0 -> the lift may be a wrapper "
                     f"artifact, not the offsets. STOP and debug.")
        print("PASS: wrapper is a behavioral identity at zero offsets.")

    elif args.mode == "control-gate":
        base = get_metric(args.base_dir, t, m)
        align = get_metric(args.align_dir, t, m)
        v7mc = get_metric(args.adapter_json, t, m)
        eff = align - base
        v7lift = v7mc - base
        frac = (eff / v7lift) if v7lift != 0 else float("inf")
        print(f"[control-gate] {t}/{m}: base={base:.4f} align={align:.4f} v7mc={v7mc:.4f} | "
              f"align_effect={eff:+.4f} v7mc_lift={v7lift:+.4f} frac={frac:.2f}")
        if abs(eff) < args.min_effect:
            sys.exit(f"FAIL (no-op): align effect {eff:+.4f} < min {args.min_effect}. The SFT "
                     f"control may be a silent no-op (vision-tower bug / undertrained) -> "
                     f"'it doesn't lift' is UNINTERPRETABLE. Verify train loss converged + retrain.")
        if frac >= args.max_frac:
            print(f"WARN: align replicates {frac:.0%} of the v7mc lift (>= {args.max_frac:.0%}). "
                  f"The contrastive-objective claim is WEAKENED -> the paper reframes toward "
                  f"'fine-tuning helps / Pareto' (data-first; a valid branch). Not a fail, a finding.")
        else:
            print(f"PASS: control is non-trivial (effect {eff:+.4f}) and its lift is << v7mc "
                  f"({frac:.0%}) -> the lift is attributable to the contrastive objective.")


if __name__ == "__main__":
    main()
