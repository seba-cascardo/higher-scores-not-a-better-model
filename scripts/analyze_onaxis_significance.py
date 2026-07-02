"""On-axis significance — paired McNemar + bootstrap CI for the on-axis causal arms.

Reviewer-blocker (audit 2026-07-02 §2-D1): the W_know recovery of 6.4% (acc 0.522 vs
null 0.500 on ARC-C n=400) is ~9 items — is it statistically distinguishable from zero?
This script answers it from the persisted per-item lm-eval samples (NO re-run, €0 local).

Arms:
  W_know  (rama A — per-item present): runs/wknow_causal/f_{null,wknow,mc}.json carry
          lm-eval `samples[arc_challenge]` with per-doc acc_norm + doc_id -> paired McNemar
          (wknow vs null) + paired bootstrap CI of recovery fraction.
  v_inf   (rama B — no per-item): runs/vinf_causal/d_{null,vinf,mc}.json are aggregate-only
          (no `samples`) -> unpaired Wilson CI of the v_inf accuracy; a paired test would
          require re-running that arm with --log_samples.

Recovery fraction:  frac = (acc_arm - acc_null) / (acc_mc - acc_null)   (acc_norm on ARC-C).

Gate (pre-registered in the remediation plan, L6-Step4):
  p_mcnemar >= 0.05 (or the accuracy CI includes 0.500) -> the paper text becomes
    "W_know recovers 6.4%, statistically indistinguishable from zero" (STRENGTHENS the
    on-axis-fails claim). p < 0.05 -> "6.4%, small but nonzero (p=...)".
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def load_accnorm(path, task="arc_challenge"):
    """lm-eval json with `samples` -> {doc_id: 0/1 acc_norm}."""
    d = json.load(open(path, encoding="utf-8"))
    if "samples" not in d or task not in d["samples"]:
        return None
    return {r["doc_id"]: int(round(r["acc_norm"])) for r in d["samples"][task]}


def agg_accnorm(path, task="arc_challenge", metric="acc_norm,none"):
    """lm-eval json (aggregate) -> scalar accuracy for the task."""
    d = json.load(open(path, encoding="utf-8"))
    res = d.get("results", {}).get(task, {})
    return res.get(metric, res.get("acc_norm,none", res.get("acc,none")))


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(center - half), float(center + half))


def mcnemar_exact(a, b, ids):
    """Exact (binomial) two-sided McNemar for two paired 0/1 dicts over ids.
    Returns b_count (a-right/b-wrong), c_count (a-wrong/b-right), p."""
    bc = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    cc = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    disc = bc + cc
    p = binomtest(min(bc, cc), disc, 0.5, alternative="two-sided").pvalue if disc > 0 else 1.0
    return bc, cc, p


def paired_bootstrap_frac(null, arm, mc, ids, n_boot=10000, seed=0):
    """Percentile 95% CI of recovery frac via paired item resampling."""
    rng = np.random.default_rng(seed)
    ids = np.array(ids)
    yn = np.array([null[i] for i in ids], float)
    ya = np.array([arm[i] for i in ids], float)
    ym = np.array([mc[i] for i in ids], float)
    n = len(ids)
    fracs = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, n, n)
        an, aa, am = yn[idx].mean(), ya[idx].mean(), ym[idx].mean()
        d = am - an
        fracs[t] = (aa - an) / d if abs(d) > 1e-9 else np.nan
        if (t + 1) % 2000 == 0:
            print(f"    [boot {t+1}/{n_boot}]", flush=True)
    fracs = fracs[~np.isnan(fracs)]
    return float(np.percentile(fracs, 2.5)), float(np.percentile(fracs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wknow-dir", default="runs/wknow_causal")
    ap.add_argument("--vinf-dir", default="runs/vinf_causal")
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="runs/wknow_causal/onaxis_significance.json")
    args = ap.parse_args()

    out = {"task": args.task, "arms": {}}

    # ---- W_know arm (rama A: per-item present) ------------------------------
    print("[wknow] loading per-item acc_norm ...", flush=True)
    null = load_accnorm(f"{args.wknow_dir}/f_null.json", args.task)
    wk = load_accnorm(f"{args.wknow_dir}/f_wknow.json", args.task)
    mc = load_accnorm(f"{args.wknow_dir}/f_mc.json", args.task)
    if null and wk and mc:
        ids = sorted(set(null) & set(wk) & set(mc))
        n = len(ids)
        a_null = sum(null[i] for i in ids) / n
        a_wk = sum(wk[i] for i in ids) / n
        a_mc = sum(mc[i] for i in ids) / n
        frac = (a_wk - a_null) / (a_mc - a_null)
        b, c, p = mcnemar_exact(null, wk, ids)  # a=null vs b=wknow
        print(f"[wknow] n={n} acc null={a_null:.4f} wknow={a_wk:.4f} mc={a_mc:.4f} "
              f"frac={frac:.4f}", flush=True)
        print(f"[wknow] McNemar wknow-vs-null b={b} c={c} p={p:.4f}", flush=True)
        print(f"[wknow] paired bootstrap CI of frac (n_boot={args.n_boot}) ...", flush=True)
        f_lo, f_hi = paired_bootstrap_frac(null, wk, mc, ids, args.n_boot)
        k_wk = int(round(a_wk * n))
        wlo, whi = wilson_ci(k_wk, n)
        out["arms"]["wknow"] = {
            "method": "paired_mcnemar_and_bootstrap", "n": n,
            "acc_null": a_null, "acc_arm": a_wk, "acc_mc": a_mc,
            "frac_recovery": frac, "frac_CI95_boot": [f_lo, f_hi],
            "mcnemar_b_nullright_armwrong": b, "mcnemar_c_nullwrong_armright": c,
            "mcnemar_discordant": b + c, "p_mcnemar": p,
            "acc_arm_wilson_CI95": [wlo, whi], "acc_arm_wilson_includes_null_0.5": bool(wlo <= 0.5 <= whi),
        }

    # ---- v_inf arm (rama B: aggregate only) ---------------------------------
    print("[vinf] aggregate accuracies (no per-item samples) ...", flush=True)
    dn = agg_accnorm(f"{args.vinf_dir}/d_null.json", args.task)
    dv = agg_accnorm(f"{args.vinf_dir}/d_vinf.json", args.task)
    dm = agg_accnorm(f"{args.vinf_dir}/d_mc.json", args.task)
    if dn is not None and dv is not None and dm is not None:
        n_vinf = 400  # d_* runs used --limit 400
        frac_v = (dv - dn) / (dm - dn)
        k_v = int(round(dv * n_vinf))
        wlo, whi = wilson_ci(k_v, n_vinf)
        print(f"[vinf] acc null={dn:.4f} vinf={dv:.4f} mc={dm:.4f} frac={frac_v:.4f}", flush=True)
        out["arms"]["vinf"] = {
            "method": "unpaired_wilson", "n": n_vinf,
            "acc_null": dn, "acc_arm": dv, "acc_mc": dm, "frac_recovery": frac_v,
            "acc_arm_wilson_CI95": [wlo, whi], "acc_arm_wilson_includes_null_0.5": bool(wlo <= 0.5 <= whi),
            "paired_test_requires": "re-run v_inf arm with --log_samples (add-on to any future pod run)",
        }

    # ---- Gate verdict -------------------------------------------------------
    wk_arm = out["arms"].get("wknow", {})
    p_wk = wk_arm.get("p_mcnemar", 1.0)
    wilson_incl = wk_arm.get("acc_arm_wilson_includes_null_0.5", True)
    if p_wk >= 0.05 or wilson_incl:
        out["gate_verdict"] = ("INDISTINGUISHABLE_FROM_ZERO: W_know recovers 6.4%, "
                               "statistically indistinguishable from zero (strengthens on-axis-fails)")
    else:
        out["gate_verdict"] = f"SMALL_BUT_NONZERO: 6.4%, small but nonzero (p_mcnemar={p_wk:.4g})"
    print(f"[gate] {out['gate_verdict']}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
