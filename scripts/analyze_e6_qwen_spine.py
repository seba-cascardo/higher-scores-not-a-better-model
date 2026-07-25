"""E6 — Qwen causal spine: is "off-axis works / on-axis fails" a property of the
contrastive PEFT, or of Gemma?

Gemma has the effect (+35pp ARC) with a causal decomposition: W_know (the strongest
on-axis reference) recovers ~6.4%, off_perp ~100%, off_par ~0, and a norm-matched
random direction inside the same orthogonal complement ~0. Qwen replicates the
EFFECT (ARC .4825 -> .6625) and the GEOMETRY (on-axis energy 2.47%, cos(v_inf,
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

import numpy as np

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

    out["kill_rules"] = verdicts
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
