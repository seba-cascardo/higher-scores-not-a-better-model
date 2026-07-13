"""P9 fork-(b) lift-matched gate — recompute from the raw per-λ artifacts (H17).

Scripts the gate that was hand-computed in runs/p9_onaxis/verdict.md (ledger B35-P9-onaxis).
Question: at a MATCHED MC lift, does the on-axis-penalty adapter retain MORE gsm8k than simply
scaling the off-axis offset DOWN to the same lift? For every λ the answer is NO (gap < 0) → the
+35 is the off-axis mass by NECESSITY (the second causal leg complementing W_know sufficiency).

  off-axis@lift = linear interpolation of the off-axis dose curve at the on-axis ARC lift
  gap           = gsm8k(on-axis) − off-axis@lift              (FAIL if < 0: on-axis retains less)
  SE            = sqrt(p_on(1-p_on)/n + p_off(1-p_off)/n)     (two independent greedy runs, n=200)

Artifacts (HF sebacascardo87/msap-p9-onaxis-20260709 → runs/p9_onaxis/); download with:
  hf download sebacascardo87/msap-p9-onaxis-20260709 --include "mc_*.json" \
    --repo-type dataset --local-dir runs/p9_onaxis    # + "gsm8k_lam*.json" + "gsm8k_dose_lam0.json"
  mc_base.json, mc_lam{5,15,50}.json, mc_dose_lam0_s{0.25,0.5,0.75}.json   (ARC acc_norm)
  gsm8k_lam{5,15,50}.json, gsm8k_dose_lam0.json                            (gsm8k acc, n=200)

  python scripts/analyze_p9_gate.py [--run-dir runs/p9_onaxis]
"""
import argparse
import json
import math
import os

GSM_N = 200
LAMBDAS = [5, 15, 50]          # on-axis penalty strengths (v_inf target)
# λ=50 is a degenerate run (eos 0.43, mean_len 357 = runaway creative-degen, §5.5): its gsm8k is
# confounded by generation blow-up, so we report the gap but not a σ (matches verdict.md).
DEGENERATE = {50}
# Ledger B35 anchors this script must reproduce on the canonical artifacts.
_LEDGER_LAM5_GAP_PP = -9.9
_LEDGER_LAM5_SIGMA = 3.8


def arc_accnorm(path):
    """ARC-Challenge acc_norm from an lm-eval results json (the MC scoring metric)."""
    return json.load(open(path, encoding="utf-8"))["results"]["arc_challenge"]["acc_norm,none"]


def gsm8k_at_scale(path, scale=1.0):
    """gsm8k accuracy at a given activation scale from a gen json (results = list of scale rows)."""
    for r in json.load(open(path, encoding="utf-8"))["results"]:
        if abs(r["scale"] - scale) < 1e-9:
            return r["acc"], int(r.get("n", GSM_N))
    raise KeyError(f"scale {scale} not found in {path}")


def dose_curve(run_dir, base_arc):
    """Off-axis dose curve as (arc_lift, gsm8k) points, ascending lift, on the 'healthy' branch
    (scale 0..0.75). Scale 1.0 is dropped on purpose: it shares lift +35.5 with scale 0.75 but its
    gsm8k is the collapsed full-dose point (0.795 vs 0.96), which would break monotone interpolation
    and is not the lift-matched control anyway (the on-axis lifts are all small)."""
    gsm = {r["scale"]: r["acc"]
           for r in json.load(open(f"{run_dir}/gsm8k_dose_lam0.json", encoding="utf-8"))["results"]}
    arc = {
        0.0: base_arc,
        0.25: arc_accnorm(f"{run_dir}/mc_dose_lam0_s0.25.json"),
        0.5: arc_accnorm(f"{run_dir}/mc_dose_lam0_s0.5.json"),
        0.75: arc_accnorm(f"{run_dir}/mc_dose_lam0_s0.75.json"),
    }
    return sorted((arc[s] - base_arc, gsm[s]) for s in arc)


def interp_offaxis(lift, curve):
    """Linear interpolation of the dose curve's gsm8k at a given ARC lift (clamped at the ends)."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    if lift <= xs[0]:
        return ys[0]
    if lift >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if lift <= xs[i]:
            t = (lift - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def se_diff(p_on, p_off, n=GSM_N):
    """SE of the gsm8k gap = two independent binomial proportions (on-axis run vs off-axis run)."""
    return math.sqrt(p_on * (1 - p_on) / n + p_off * (1 - p_off) / n)


def gate_point(arc_lift, gsm_on, curve, n=GSM_N):
    """The lift-matched gate for one on-axis point. Returns the off-axis gsm8k at the matched
    lift, the gap (FAIL if < 0), its SE, and σ = |gap|/SE."""
    off = interp_offaxis(arc_lift, curve)
    gap = gsm_on - off
    se = se_diff(gsm_on, off, n)
    return {"off_at_lift": off, "gap": gap, "se": se,
            "sigma": abs(gap) / se if se > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/p9_onaxis")
    ap.add_argument("--out", default=None, help="output json (default: <run-dir>/p9_gate.json)")
    a = ap.parse_args()
    run = a.run_dir
    out_path = a.out or f"{run}/p9_gate.json"

    base_arc = arc_accnorm(f"{run}/mc_base.json")
    curve = dose_curve(run, base_arc)
    print(f"[p9-gate] base ARC acc_norm = {base_arc:.4f}; off-axis dose curve (lift, gsm8k):",
          flush=True)
    for lift, g in curve:
        print(f"    lift {lift * 100:+7.2f}pp  gsm8k {g:.3f}", flush=True)

    print(f"\n[p9-gate] lift-matched gate (gsm8k n={GSM_N}):", flush=True)
    print(f"  {'lam':>3} {'ARC_lift':>10} {'gsm_on':>7} {'off@lift':>9} {'gap':>9} "
          f"{'SE':>6} {'sigma':>6}  verdict", flush=True)
    rows = []
    for i, lam in enumerate(LAMBDAS, 1):
        arc_on = arc_accnorm(f"{run}/mc_lam{lam}.json")
        lift = arc_on - base_arc
        gsm_on, n = gsm8k_at_scale(f"{run}/gsm8k_lam{lam}.json")
        g = gate_point(lift, gsm_on, curve, n)
        deg = lam in DEGENERATE
        sig_s = "  —  " if deg else f"{g['sigma']:5.1f}"
        tag = "FAIL (degenerate)" if deg else "FAIL"
        print(f"  {lam:>3} {lift * 100:>+9.2f}pp {gsm_on:>7.3f} {g['off_at_lift']:>9.3f} "
              f"{g['gap'] * 100:>+8.2f}pp {g['se'] * 100:>5.1f} {sig_s:>6}  {tag}",
              flush=True)
        rows.append({"lambda": lam, "arc_lift": lift, "gsm_on": gsm_on,
                     "off_at_lift": g["off_at_lift"], "gap": g["gap"], "se": g["se"],
                     "sigma": None if deg else g["sigma"], "degenerate": deg, "verdict": "FAIL"})

    all_fail = all(r["gap"] < 0 for r in rows)
    print(f"\n[p9-gate] VERDICT: gate {'FAILS at every lambda' if all_fail else 'PASSES somewhere'}"
          " -> on-axis is Pareto-dominated by scaling the off-axis offset down (fork (b), B35).",
          flush=True)

    # Self-check vs ledger B35 / verdict.md: λ=5 gap ≈ −9.9pp, σ ≈ 3.8 (the load-bearing anchor).
    r5 = next(r for r in rows if r["lambda"] == 5)
    assert r5["gap"] < 0, "gate must FAIL at lambda=5"
    assert abs(r5["gap"] * 100 - _LEDGER_LAM5_GAP_PP) < 0.5, \
        f"lambda=5 gap {r5['gap'] * 100:.2f}pp != ledger {_LEDGER_LAM5_GAP_PP}pp"
    assert abs(r5["sigma"] - _LEDGER_LAM5_SIGMA) < 0.3, \
        f"lambda=5 sigma {r5['sigma']:.2f} != ledger {_LEDGER_LAM5_SIGMA}"
    print(f"[p9-gate] self-check vs ledger B35 OK: lambda=5 gap "
          f"{r5['gap'] * 100:+.1f}pp / sigma {r5['sigma']:.1f}", flush=True)

    json.dump({"base_arc": base_arc, "dose_curve": curve, "gsm8k_n": GSM_N,
               "gate": rows, "conclusion": "fork_b_fail_all_lambda"},
              open(out_path, "w"), indent=2)
    print(f"[p9-gate] wrote {os.path.relpath(out_path)}", flush=True)


if __name__ == "__main__":
    main()
