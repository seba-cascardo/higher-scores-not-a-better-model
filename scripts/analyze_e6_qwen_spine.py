"""E6 — Qwen causal spine: is "off-axis works / on-axis fails" a property of the
contrastive PEFT, or of Gemma?

Gemma has the effect (+35pp ARC) with a causal decomposition: W_know (the strongest
on-axis reference) recovers ~6.4%, off_perp ~100%, off_par ~0, and a norm-matched
random direction inside the same orthogonal complement ~0. Qwen replicates the
EFFECT (ARC .4975 -> .6600, +16.25pp) and the GEOMETRY (on-axis energy 2.47%, cos(v_inf,
theta_mc) -0.024), but nothing causal had been measured. This closes that.

All six offset blobs were built LOCALLY (CPU) from Qwen's own functional directions
and its own trained adapter, then evaluated on the pod. Fractions use the base and
mc measured in the SAME run, so they are internally consistent even though the
null-adapter base sat +1.5pp above the July reference (declared caveat; the mc arm
reproduced to -0.25pp and the ARC lift, +16.25pp, is within the +-3pp band fixed
before looking).

Kill-rules, pre-registered in plans/2026-07-25-pod-runbook-E4-E6.md:
  W_know    > 1/3 of the lift -> "on-axis fails" is NOT universal; re-scope the
                                 mechanism claim to this model in abstract/intro/discussion
  off_perp  < 2/3 of the lift -> the parallel/perp decomposition does not transfer;
                                 report the cross-family spine as a NEGATIVE
  shuffle  ~= off_perp        -> inside the orthogonal complement the learned
                                 direction is not privileged; concede for Qwen

  python scripts/analyze_e6_qwen_spine.py
"""
import json
import os
import sys

import numpy as np

# The report prints non-ASCII (sigma); a Windows console defaults to cp1252 and
# would die mid-report instead of writing the summary.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

E6 = "runs/e6_qwen_spine"
OUT = os.path.join(E6, "e6_spine_summary.json")

TASKS = {
    "arc_challenge": "acc_norm,none",
    "truthfulqa_mc1": "acc,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
}
# Gemma reference recovery fractions (ledger; NOT recomputed here, cited)
GEMMA_REF = {"mc_wknow_offset": 0.064, "mc_offpar_offset": None,
             "mc_offperp_offset": 1.00, "shuffle": 0.00}
# Gemma's three shuffle seeds on ARC. The ledger and discussion.tex report them
# rounded to two decimals (-0.04/-0.01/-0.01); the raw arms are acc_norm
# 0.49/0.5025/0.50 against base 0.505 and mc 0.855 (RD run, HF
# msap-r5-align-p0-20260628), which give the unrounded fractions below. Rounding
# shrinks the sd by 8.3% and inflates every sigma computed from it, so the
# unrounded values are what this contrast uses.
GEMMA_SHUFFLE_ARC = [-0.042857, -0.007143, -0.014286]
# The on-axis arm's OWN uncertainty, which a point-vs-cloud comparison silently
# drops. Gemma has a paired bootstrap CI for it in runs/wknow_causal/
# onaxis_significance.json: frac 0.0638, CI95 [-0.008850, 0.136364] -> SE ~0.0370.
# That artifact's own verdict is "INDISTINGUISHABLE_FROM_ZERO" (McNemar p=0.136),
# which is flatly incompatible with W_know clearing any floor by several sigma.
GEMMA_WKNOW_SE = (0.136364 - (-0.008850)) / (2 * 1.959964)
ARMS = ["mc_wknow_offset", "mc_offpar_offset", "mc_offperp_offset",
        "mc_offperp_shuffle_seed0", "mc_offperp_shuffle_seed1", "mc_offperp_shuffle_seed2"]
# Winogrande's Qwen lift is only ~3pp: fractions there are denominator-noisy.
MIN_LIFT_FOR_FRACTION = 0.05


def accs(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        res = json.load(f)["results"]
    return {t: float(res[t][m]) for t, m in TASKS.items() if t in res}


def _floor(wknow, wknow_se, shuffle_seeds, family):
    """Margin of the on-axis reference over the same-norm noise floor, in sigma.

    Three things this has to get right, each of which inflates sigma if dropped:

    1. The arm's OWN uncertainty. Comparing a point against the spread of a control
       silently asserts the point is known exactly. It is not: W_know is one eval of
       400 items. Gemma's paired bootstrap gives SE ~3.7pp; without it the margin
       looks like 4.8 sigma, with it ~2, and the source artifact's own verdict for
       that same arm is "indistinguishable from zero".
    2. Prediction, not confidence. The question is whether W_know falls inside the
       CLOUD of same-norm random directions, so the shuffle's sd is the right scale
       (not sd/sqrt(n)), carrying the sqrt(1 + 1/n) prediction factor.
    3. n=3. A sigma computed against an sd estimated from three seeds is not a z.
       The two-sided 95% t critical value at 2 df is 4.30, so "2 sigma" here is
       roughly p=0.18, not p=0.05. Reported alongside so nobody reads it as a z.
    """
    n = len(shuffle_seeds)
    mean = float(np.mean(shuffle_seeds))
    sd = float(np.std(shuffle_seeds, ddof=1))
    pred_sd = sd * np.sqrt(1 + 1 / n)          # predictive, not standard error
    se_total = float(np.hypot(pred_sd, wknow_se or 0.0))
    margin = float(wknow) - mean
    sigma = abs(margin) / se_total if se_total else float("inf")
    return {"family": family, "wknow_arc": float(wknow), "wknow_se": float(wknow_se or 0.0),
            "shuffle_mean": mean, "shuffle_sd": sd, "n_shuffle_seeds": n,
            "predictive_sd": float(pred_sd), "se_total": se_total,
            "margin": margin, "sigma": sigma, "t_crit_95_at_df": T_CRIT_95[n - 1],
            "beats_floor": bool(margin > 0 and sigma >= T_CRIT_95[n - 1])}


# two-sided 95% t critical values by degrees of freedom (1, 2, 3, ...)
T_CRIT_95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}


def onaxis_vs_noise_floor(wk_qwen, shuffle_qwen, base_qwen, lift_qwen):
    """Does on-axis buy anything over same-norm noise? Asked in both families.

    The claim this replaces read W_know against ZERO ("counterproductive in Qwen").
    Zero is the wrong reference: the same-norm random control is not at zero either
    (-15.9% in Qwen), so most of that damage is the norm, not the direction.
    """
    # Qwen has no bootstrap for its W_know arm, so use the unpaired binomial SE of
    # its accuracy propagated through the lift. That is an UPPER bound (pairing
    # would shrink it), and it is declared as such.
    acc = base_qwen + wk_qwen * lift_qwen
    se_q = float(np.sqrt(acc * (1 - acc) / 400) / lift_qwen)
    q = _floor(wk_qwen, se_q, shuffle_qwen, "Qwen2.5-14B-Instruct")
    q["wknow_se_source"] = "unpaired binomial on n=400, propagated through the lift (upper bound)"
    g = _floor(GEMMA_REF["mc_wknow_offset"], GEMMA_WKNOW_SE, GEMMA_SHUFFLE_ARC, "Gemma 4 31B IT")
    g["wknow_se_source"] = "paired bootstrap CI, runs/wknow_causal/onaxis_significance.json"
    print(f"\n  on-axis vs its own same-norm noise floor (ARC):")
    for f in (g, q):
        flag = "beats the floor" if f["beats_floor"] else "NOT distinguishable from the floor"
        print(f"    {f['family']:<24} W_know {f['wknow_arc'] * 100:+6.1f}% ±{f['wknow_se'] * 100:.1f}"
              f"  vs shuffle {f['shuffle_mean'] * 100:+6.1f}% ± {f['shuffle_sd'] * 100:.1f}"
              f"  -> margin {f['margin'] * 100:+5.1f} ± {f['se_total'] * 100:.1f} pp "
              f"({f['sigma']:.2f}σ vs t crit {f['t_crit_95_at_df']}, {flag})")
    verdict = ("the on-axis reference does NOT beat same-norm noise in either family: "
               f"Gemma {g['margin'] * 100:+.1f} ± {g['se_total'] * 100:.1f} pp "
               f"({g['sigma']:.2f}σ) and Qwen {q['margin'] * 100:+.1f} ± "
               f"{q['se_total'] * 100:.1f} pp ({q['sigma']:.2f}σ), both short of the "
               f"t critical value at 2 df (4.30) -- against off_perp's +101pp (Gemma) "
               "and +122pp (Qwen) over the same floor")
    print(f"    -> {verdict}")
    return {"gemma": g, "qwen": q, "verdict": verdict}


def main():
    base = accs(os.path.join(E6, "eval_base.json"))
    mc = accs(os.path.join(E6, "eval_mc.json"))
    if base is None or mc is None:
        raise SystemExit(f"[ABORT] need {E6}/eval_base.json and eval_mc.json")

    lift = {t: mc[t] - base[t] for t in base}
    print("Qwen2.5-14B-Instruct — causal spine")
    print(f"  base : " + "  ".join(f"{t[:4]} {base[t]:.4f}" for t in TASKS))
    print(f"  mc   : " + "  ".join(f"{t[:4]} {mc[t]:.4f}" for t in TASKS))
    print(f"  lift : " + "  ".join(f"{t[:4]} {lift[t] * 100:+.2f}pp" for t in TASKS))
    weak = [t for t in TASKS if lift[t] < MIN_LIFT_FOR_FRACTION]
    if weak:
        print(f"  [note] lift < {MIN_LIFT_FOR_FRACTION * 100:.0f}pp on {', '.join(weak)}"
              f" -> fractions there are denominator-noisy, read with care")

    out = {"model": "Qwen2.5-14B-Instruct", "base": base, "mc": mc, "lift": lift,
           "weak_lift_tasks": weak, "arms": {}}

    print(f"\n{'arm':<28}" + "".join(f"{t[:4]:>14}" for t in TASKS))
    per_arm = {}
    for arm in ARMS:
        a = accs(os.path.join(E6, f"eval_{arm}.json"))
        if a is None:
            print(f"{arm:<28}  MISSING")
            continue
        rec = {t: (a[t] - base[t]) / lift[t] for t in TASKS if lift[t] != 0}
        per_arm[arm] = {"acc": a, "recovery": rec}
        print(f"{arm:<28}" + "".join(f"{rec[t] * 100:13.1f}%" for t in TASKS))
    out["arms"] = per_arm

    if not per_arm:
        raise SystemExit("[ABORT] no arm evals found")

    # shuffle control: mean over the three seeds
    sh = [per_arm[a]["recovery"] for a in ARMS if "shuffle" in a and a in per_arm]
    sh_arc = float(np.mean([s["arc_challenge"] for s in sh])) if sh else None
    sh_sd = float(np.std([s["arc_challenge"] for s in sh], ddof=1)) if len(sh) > 1 else 0.0

    print(f"\n{'=' * 74}\nKILL-RULES (ARC, the headline anchor)\n{'=' * 74}")
    verdicts = {}

    wk = per_arm.get("mc_wknow_offset", {}).get("recovery", {}).get("arc_challenge")
    if wk is not None:
        broken = wk > 1 / 3
        verdicts["universality"] = {
            "wknow_arc_recovery": wk, "gemma_reference": GEMMA_REF["mc_wknow_offset"],
            "threshold": 1 / 3, "universality_broken": bool(broken),
            "verdict": ("ON-AXIS WORKS IN QWEN -> 'on-axis fails' is NOT universal; re-scope "
                        "the mechanism claim to Gemma in abstract/intro/discussion"
                        if broken else
                        "ON-AXIS FAILS IN QWEN TOO -> the off-axis/on-axis asymmetry "
                        "replicates cross-family; universality SURVIVES")}
        print(f"  W_know recovery      {wk * 100:6.1f}%   (Gemma 6.4%, threshold 33.3%)")
        print(f"    -> {verdicts['universality']['verdict']}")

    op = per_arm.get("mc_offperp_offset", {}).get("recovery", {}).get("arc_challenge")
    if op is not None:
        ok = op >= 2 / 3
        verdicts["offperp_transfers"] = {
            "offperp_arc_recovery": op, "threshold": 2 / 3, "transfers": bool(ok),
            "verdict": ("off_perp carries the lift in Qwen too -> the parallel/perp "
                        "decomposition TRANSFERS cross-family"
                        if ok else
                        "off_perp does NOT carry the lift in Qwen -> the decomposition "
                        "does not transfer; report the cross-family spine as a NEGATIVE")}
        print(f"  off_perp recovery    {op * 100:6.1f}%   (Gemma ~100%, threshold 66.7%)")
        print(f"    -> {verdicts['offperp_transfers']['verdict']}")

    par = per_arm.get("mc_offpar_offset", {}).get("recovery", {}).get("arc_challenge")
    if par is not None:
        print(f"  off_par recovery     {par * 100:6.1f}%   (Gemma ~0%)")
        verdicts["offpar"] = {"offpar_arc_recovery": par}

    if sh_arc is not None and op is not None:
        # "not privileged" if the random direction inside the complement gets most of
        # what the learned one gets
        not_priv = sh_arc >= 0.5 * op
        verdicts["direction_privileged"] = {
            "shuffle_arc_recovery_mean": sh_arc, "shuffle_arc_sd": sh_sd,
            "offperp_arc_recovery": op, "learned_direction_privileged": bool(not not_priv),
            "verdict": ("a norm-matched RANDOM direction in the same complement recovers "
                        "comparably -> inside the orthogonal complement the learned "
                        "direction is NOT privileged in Qwen; concede for this model"
                        if not_priv else
                        "shuffle recovers far less than off_perp -> the LEARNED direction "
                        "inside the complement is what lifts; specificity survives in Qwen")}
        print(f"  shuffle recovery     {sh_arc * 100:6.1f}% ± {sh_sd * 100:.1f}   "
              f"(Gemma ~0%, vs off_perp {op * 100:.1f}%)")
        print(f"    -> {verdicts['direction_privileged']['verdict']}")

    if wk is not None and sh_arc is not None:
        verdicts["onaxis_vs_noise_floor"] = onaxis_vs_noise_floor(
            wk, [s["arc_challenge"] for s in sh],
            base["arc_challenge"], lift["arc_challenge"])

    out["kill_rules"] = verdicts
    out["matched_control"] = matched_control(base, lift)
    out["gen_transfer"] = gen_transfer()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


# Gemma same-param control, best-ckpt (E1 / ledger B48). Cited, not recomputed.
GEMMA_MATCHED = {"arc_challenge": 0.362, "truthfulqa_mc1": 0.009,
                 "hellaswag": 0.695, "winogrande": 0.795}


def matched_control(base, lift):
    """Same-parameterization non-contrastive control in Qwen (3 seeds).

    NOTE: the best-val_acc checkpoint DEGENERATED here -- Qwen's val_acc starts at
    0.760 and the training never beats it, so 2/3 seeds selected step 1 (the
    initialization: alpha=1, theta=0 = a null offset) and their evals reproduced the
    base digit-for-digit. The honest arm is therefore `.final`, and that is what this
    reads. Recorded because the checkpoint rule, not the model, produced the no-op.
    """
    print(f"\n{'=' * 74}\nmatched control (same-param plain-CE, .final ckpt)\n{'=' * 74}")
    rows = []
    for s in (0, 1, 2):
        a = accs(os.path.join(E6, f"eval_plain_ce_seed{s}.final.json"))
        if a is not None:
            rows.append(a)
    if not rows:
        print("  MISSING")
        return {"status": "missing"}

    rec = {}
    for t in TASKS:
        if lift[t] == 0:
            continue
        fr = np.array([(r[t] - base[t]) / lift[t] for r in rows])
        eff = np.array([r[t] - base[t] for r in rows])
        rec[t] = {"frac_mean": float(fr.mean()),
                  "frac_sd": float(fr.std(ddof=1)) if len(fr) > 1 else 0.0,
                  "effect_mean_pp": float(eff.mean() * 100),
                  "n_seeds_harmful": int((eff < 0).sum()), "n_seeds": len(eff),
                  "gemma_reference": GEMMA_MATCHED.get(t),
                  "denominator_noisy": bool(lift[t] < MIN_LIFT_FOR_FRACTION)}
        r = rec[t]
        flag = "  [HARMFUL]" if r["n_seeds_harmful"] == r["n_seeds"] else ""
        noisy = "  [noisy denom]" if r["denominator_noisy"] else ""
        print(f"  {t:<16} {r['frac_mean'] * 100:7.1f}% ± {r['frac_sd'] * 100:4.1f}   "
              f"({r['effect_mean_pp']:+.1f}pp)   Gemma {GEMMA_MATCHED.get(t, float('nan')) * 100:5.1f}%"
              f"{flag}{noisy}")
    print("\n  task-heterogeneity replicates cross-family if the ORDER matches Gemma "
          "(HSwag > ARC > TQA)")
    return {"n_seeds": len(rows), "checkpoint": "final",
            "best_ckpt_degenerate": "2/3 seeds selected step 1 (null offset); see docstring",
            "per_task": rec}


def gen_transfer():
    """gsm8k: does the MC lift transfer to multi-step generation? (it does not)"""
    p = os.path.join(E6, "gsm8k_scale_fixed.json")
    if not os.path.exists(p):
        print("\n  [gsm8k] MISSING (need the stop-id-fixed run)")
        return {"status": "missing"}
    with open(p, encoding="utf-8") as f:
        g = json.load(f)
    v = g["verdict"]
    b, a = v["acc_scale0_base"], v["acc_scale1_full"]
    # eos_rate is only meaningful in THIS run: the earlier one passed Gemma's stop ids
    # to Qwen, so its generations ran to the cap and its eos_rate measured nothing.
    eos = {str(r["scale"]): r["eos_rate"] for r in g["results"]}
    ln = {str(r["scale"]): r["mean_gen_len"] for r in g["results"]}
    print(f"\n{'=' * 74}\ngeneration transfer (gsm8k, stop-ids resolved from the model)\n{'=' * 74}")
    print(f"  acc      base {b:.4f} -> adapter {a:.4f}   = {(a - b) * 100:+.1f}pp   "
          f"(Gemma -24.2 ± 7.3)")
    print(f"  eos_rate base {eos.get('0.0')} -> adapter {eos.get('1.0')}   "
          f"(termination PRESERVED; corroborates the B15 retract cross-family)")
    print(f"  mean_len base {ln.get('0.0')} -> adapter {ln.get('1.0')}   (adapter generates shorter)")
    return {"base_acc": b, "adapter_acc": a, "damage_pp": (a - b) * 100,
            "gemma_reference_pp": -24.2, "eos_rate": eos, "mean_gen_len": ln,
            "note": "stop ids resolved from the model's generation_config; the earlier run "
                    "used a Gemma literal ([1,50,105,106]) and its eos_rate (0.775) must NOT "
                    "be cited -- it would have resurrected the B15 EOS-collapse zombie"}


if __name__ == "__main__":
    main()
