"""On-axis ENERGY BUDGET (energy-weighted) + formal random null for the NET-EFFECT
footprints of the contrastive (LoFiT) vs Align (generic SFT) adapters. €0-local.

Reuses the already-computed pod footprints runs/derisk/{contrastive,align}_direction.json
(each row: per-head cos(diff, v_inf) and diff_norm, where diff = act_treated - act_base
captured on ARC). The raw theta offset is a DIFFERENT object (decomposed separately in
analyze_offset_parperp_decomposition.py); the net effect is the apples-to-apples object
for contrastive-vs-SFT (SFT has no per-head raw offset).

Metric upgrade over B6b/B6g (which reported mean|cos|/mean|cos_rand|): here we report the
energy-weighted on-axis budget  B = sum_h(diff_norm_h^2 * cos_h^2) / sum_h(diff_norm_h^2)
(= fraction of the net effect's total energy projecting onto v_inf), and compare it to a
formal random-direction null:  cos^2 of a random dir in R^d ~ Beta(1/2,(d-1)/2),
E[cos^2]=1/d. Null budget keeps the SAME per-head weights diff_norm^2.

  EXPLORATORY (1 rank/task, v_inf only, net-effect not raw). NOT draft material.

  python scripts/analyze_neteffect_onaxis_budget.py
"""
import json
import os

import numpy as np

ROOT = os.environ.get("MSAP_ROOT", ".")
FILES = {
    "contrastive (LoFiT)": os.path.join(ROOT, "runs/derisk/contrastive_direction.json"),
    "align (generic SFT)": os.path.join(ROOT, "runs/derisk/align_direction.json"),
}
HEAD_DIM = 256
K_NULL = 5000
OUT = os.path.join(ROOT, "runs/derisk/neteffect_onaxis_budget.json")


def analyze(rows, rng):
    dn2 = np.array([r["diff_norm"] ** 2 for r in rows], dtype=float)
    cos = np.array([r["cos_vinf"] for r in rows], dtype=float)
    W = dn2 / dn2.sum()
    budget = float((W * cos ** 2).sum())            # energy-weighted on-axis budget
    budget_unw = float((cos ** 2).mean())            # unweighted
    # formal null: cos^2 of a random dir ~ Beta(1/2,(d-1)/2), same per-head weights
    null = rng.beta(0.5, (HEAD_DIM - 1) / 2.0, size=(K_NULL, len(rows)))
    null_budgets = (null * W).sum(axis=1)
    nmean, nstd = float(null_budgets.mean()), float(null_budgets.std())
    p95, p99 = float(np.percentile(null_budgets, 95)), float(np.percentile(null_budgets, 99))
    z = (budget - nmean) / nstd if nstd > 0 else float("nan")
    pctile = float((null_budgets < budget).mean() * 100.0)
    # also the B6 ratio for continuity
    ratio_abscos = float(np.abs(cos).mean() /
                         max(np.mean([r["mean_abs_cos_rand"] for r in rows]), 1e-9))
    return {
        "n_heads": len(rows),
        "on_axis_energy_budget_vinf": budget,
        "budget_unweighted": budget_unw,
        "vs_random_null": {"null_mean": nmean, "expected_1_over_d": 1.0 / HEAD_DIM,
                           "null_p95": p95, "null_p99": p99,
                           "ratio_to_null_mean": budget / nmean, "z_score": z,
                           "percentile_in_null": pctile,
                           "above_null_p95": bool(budget > p95)},
        "b6_ratio_mean_abscos": ratio_abscos,
    }


def main():
    rng = np.random.default_rng(0)
    print("=" * 72)
    print("net-effect on-axis ENERGY BUDGET vs v_inf + random null  (contrastive vs SFT)")
    print("=" * 72)
    res = {}
    for label, path in FILES.items():
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        d = json.load(open(path))
        res[label] = analyze(d["rows"], rng)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"head_dim": HEAD_DIM, "k_null": K_NULL, "results": res}, open(OUT, "w"), indent=2)

    print(f"\n  random-dir null budget: mean ~ 1/d = {1.0/HEAD_DIM*100:.3f}%  (K={K_NULL})\n")
    print(f"  {'adapter':22s} {'budget':>8s} {'xNull':>6s} {'z':>7s} {'pctile':>7s} "
          f"{'>p95?':>6s} {'B6_ratio|cos|':>13s}")
    for label, r in res.items():
        n = r["vs_random_null"]
        print(f"  {label:22s} {r['on_axis_energy_budget_vinf']*100:7.2f}% "
              f"{n['ratio_to_null_mean']:5.2f}x {n['z_score']:+7.2f} "
              f"{n['percentile_in_null']:6.1f}% {str(n['above_null_p95']):>6s} "
              f"{r['b6_ratio_mean_abscos']:12.2f}x")
    print(f"\n  READ: both small => neither lives on-axis. SFT budget >= contrastive => "
          f"off-axis is NOT a contrastive signature (off-axis of degree). EXPLORATORY: "
          f"1 rank/task, v_inf only, net-effect. NOT draft material.")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
