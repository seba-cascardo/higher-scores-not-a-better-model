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
    ap.add_argument("--base-align-dir",
                    help="control-gate: base measured in the SAME harness as --align-dir "
                         "(native HFLM). Used for the NUMERATOR (align_effect) so it does not "
                         "cross harnesses with the v7mc denominator. Defaults to --base-dir.")
    ap.add_argument("--adapter-json")
    ap.add_argument("--align-dir")
    ap.add_argument("--expect-base", type=float)
    ap.add_argument("--expect-adapter", type=float)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-effect", type=float, default=0.01)
    ap.add_argument("--frac-lo", type=float, default=1.0 / 3,
                    help="control-gate zone A/C boundary (PASS if frac < frac-lo). Default 1/3.")
    ap.add_argument("--frac-hi", type=float, default=2.0 / 3,
                    help="control-gate zone C/B boundary (WARN if frac >= frac-hi). Default 2/3.")
    ap.add_argument("--max-frac", type=float, default=None,
                    help="DEPRECATED single-cut alias. If set, collapses to a 2-zone gate at this "
                         "value (sets frac-lo=frac-hi=max-frac); prefer --frac-lo/--frac-hi.")
    args = ap.parse_args()
    if args.max_frac is not None:
        args.frac_lo = args.frac_hi = args.max_frac

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
        # NUMERATOR (align_effect): align and its base measured in the SAME (native HFLM) harness.
        # DENOMINATOR (v7lift): v7mc and base measured in the SAME (V7HFLM) harness. Keeping the two
        # ratios harness-internal removes the cross-harness bias that would confound `frac`.
        base_num = get_metric(args.base_align_dir or args.base_dir, t, m)  # native HFLM base
        base_den = get_metric(args.base_dir, t, m)                          # V7HFLM base (d_null)
        align = get_metric(args.align_dir, t, m)
        v7mc = get_metric(args.adapter_json, t, m)
        eff = align - base_num
        v7lift = v7mc - base_den
        print(f"[control-gate] {t}/{m}: base_num={base_num:.4f} align={align:.4f} "
              f"base_den={base_den:.4f} v7mc={v7mc:.4f} | "
              f"align_effect={eff:+.4f} v7mc_lift={v7lift:+.4f}")
        # 0) SETUP gate: the canonical +35 must actually exist (positive, non-trivial) on this
        #    task/path, else `frac` (fraction OF that lift) is meaningless. Guards v7lift<=0 and
        #    the near-zero blow-up the bare `v7lift != 0` check missed.
        if v7lift <= args.min_effect:
            sys.exit(f"FAIL (setup): v7mc lift {v7lift:+.4f} <= min {args.min_effect} on {t}. "
                     f"The +35 is not present on this task/path -> the control-gate fraction is "
                     f"undefined. Check base/v7mc paths or run on the task where the lift exists "
                     f"(arc_challenge).")
        frac = eff / v7lift  # v7lift > 0 guaranteed here
        # 1) CONVERGENCE gate: signed (NOT abs). A large NEGATIVE effect is a broken/harmful control,
        #    not "converged to a small lift" -> FAIL, never PASS.
        if eff < args.min_effect:
            sys.exit(f"FAIL (no-op-or-harmful): align effect {eff:+.4f} < min {args.min_effect}. "
                     f"The SFT control is a silent no-op OR HURT accuracy (eff<0) -> 'it doesn't "
                     f"lift' is UNINTERPRETABLE. Verify train loss converged + retrain.")
        # 2) Three-zone reading (pre-reg A/B/C). eff>0 and v7lift>0 -> frac in (0, inf), so the
        #    printed % is always meaningful. Thresholds anchored on frac (1/3, 2/3), NOT absolute
        #    align_effect, so they hold even if the measured v7lift differs from +0.350.
        lo, hi = args.frac_lo, args.frac_hi
        if frac < lo:
            print(f"PASS (zone A): align replicates {frac:.0%} of the v7mc lift (< {lo:.0%}). "
                  f"The matched SFT does NOT reproduce the +35 -> the lift is attributable to the "
                  f"contrastive objective. Thesis intact.")
        elif frac >= hi:
            print(f"WARN (zone B): align replicates {frac:.0%} of the v7mc lift (>= {hi:.0%}). "
                  f"The contrastive-objective claim is WEAKENED -> the paper reframes toward "
                  f"'fine-tuning helps / Pareto' (data-first; a valid branch). Not a fail, a finding.")
        else:
            print(f"MIXED (zone C): align replicates {frac:.0%} of the v7mc lift "
                  f"([{lo:.0%}, {hi:.0%})). Partial reproduction -> report the honest fraction; "
                  f"neither a clean intact-thesis (A) nor a clean reframe (B).")


if __name__ == "__main__":
    main()
