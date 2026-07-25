"""E4 — 2x2 factorial: training OBJECTIVE (contrastive vs plain-CE) x DOMAIN
(in-domain TQA+ARC vs OOD-matched MedMCQA).

Closes the confound the paper declares open (limitations.tex): the inducibility
control trains on the same corpus the effect is measured on, so "ordinary
fine-tuning of ANY kind recovers most of the lift" and "fine-tuning ON THIS DOMAIN
does" are not separated. The OOD cells separate them.

Two of the four cells already existed and are NOT recomputed:
  A contrastive x in-domain = the canonical V7-mc adapter
  B plain-CE   x in-domain = E1 (3 seeds)
  C contrastive x OOD      = runs/e4_factorial/ood_direct_seed{0,1,2}
  D plain-CE   x OOD       = runs/e4_factorial/ood_plain_ce_seed{0,1,2}

Recovery fraction per arm: frac = (arm - base) / (v7mc - base), with the SAME
apples-to-apples base and contrastive denominator E1 used.

Kill-rules are pre-registered in docs/superpowers/plans/2026-07-25-pod-runbook-E4-E6.md
and applied verbatim here (binning at 1/3 and 2/3, the paper's own thresholds):
  rec(D) >= 2/3 * rec(B)  -> "any fine-tuning": domain is not what matters
  rec(D) <= 1/3 * rec(B)  -> "domain exposure": re-scope "generic" across the paper
  in between              -> partial ambiguity, report the number, pick no reading

  python scripts/analyze_e4_factorial.py
"""
import json
import os

import numpy as np

E1 = "runs/e1_plain_ce"
E4 = "runs/e4_factorial"
OUT = os.path.join(E4, "e4_factorial_summary.json")

# task -> metric key in lm-eval results
TASKS = {
    "arc_challenge": "acc_norm,none",
    "truthfulqa_mc1": "acc,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
}
CONVERGENCE_MIN_EFFECT = 0.01     # the paper's own gate: arm - base >= 0.01
SETUP_MIN_EFFECT = 0.01           # and the contrastive lift must be real and positive


def accs(path):
    """{task: metric} from an lm-eval results JSON, or None if absent."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        res = json.load(f)["results"]
    return {t: float(res[t][m]) for t, m in TASKS.items() if t in res}


def cell(paths, label):
    """Load a cell's seeds; returns (list of acc-dicts, list of missing paths)."""
    got, missing = [], []
    for p in paths:
        a = accs(p)
        (got if a is not None else missing).append(a if a is not None else p)
    return got, missing


def main():
    base = accs(os.path.join(E1, "eval_base.json"))
    v7 = accs(os.path.join(E1, "eval_contrastive_canonical.json"))
    if base is None or v7 is None:
        raise SystemExit("[ABORT] missing base / contrastive-canonical reference evals "
                         "(pull runs/e1_plain_ce/eval_*.json from HF "
                         "msap-review-remediation-20260724)")

    # --- setup gate: the contrastive lift must be real and positive per task ---
    lift = {t: v7[t] - base[t] for t in base}
    for t, l in lift.items():
        if l < SETUP_MIN_EFFECT:
            print(f"  [warn] setup gate: {t} contrastive lift {l:+.4f} < {SETUP_MIN_EFFECT} "
                  f"-> fraction not meaningful for this task")

    cells = {}
    for ck, suffix in (("best", ""), ("final", ".final")):
        cells[ck] = {
            "B_plainCE_indomain": [os.path.join(E1, f"eval_lifts_seed{s}{suffix}.json") for s in (0, 1, 2)],
            "C_contrastive_OOD": [os.path.join(E4, f"eval_ood_direct_seed{s}{suffix}.json") for s in (0, 1, 2)],
            "D_plainCE_OOD": [os.path.join(E4, f"eval_ood_plain_ce_seed{s}{suffix}.json") for s in (0, 1, 2)],
        }

    out = {"base": base, "v7mc_contrastive_indomain": v7, "lift": lift,
           "gates": {"setup_min_effect": SETUP_MIN_EFFECT,
                     "convergence_min_effect": CONVERGENCE_MIN_EFFECT},
           "checkpoints": {}}

    for ck, spec in cells.items():
        print(f"\n{'=' * 74}\ncheckpoint: {ck}\n{'=' * 74}")
        ck_out = {}
        for name, paths in spec.items():
            got, missing = cell(paths, name)
            if not got:
                print(f"  {name:<24} MISSING ({len(missing)} files not found yet)")
                ck_out[name] = {"status": "missing", "missing": missing}
                continue

            rec = {}
            for t in TASKS:
                if t not in base or lift.get(t, 0) < SETUP_MIN_EFFECT:
                    continue
                arm = np.array([g[t] for g in got if t in g])
                eff = arm - base[t]
                noop = eff < CONVERGENCE_MIN_EFFECT
                fr = eff / lift[t]
                rec[t] = {
                    "arm_mean": float(arm.mean()), "arm_sd": float(arm.std(ddof=1)) if len(arm) > 1 else 0.0,
                    "effect_mean_pp": float(eff.mean() * 100),
                    "frac_mean": float(fr.mean()), "frac_sd": float(fr.std(ddof=1)) if len(fr) > 1 else 0.0,
                    "n_seeds": int(len(arm)),
                    "n_seeds_noop": int(noop.sum()),
                    "converged": bool((~noop).all()),
                }
            ck_out[name] = {"status": "ok" if not missing else "partial",
                            "n_seeds_found": len(got), "missing": missing, "per_task": rec}

            head = f"  {name:<24}"
            for t in TASKS:
                if t in rec:
                    r = rec[t]
                    flag = "" if r["converged"] else "  [NO-OP]"
                    head += f"  {t[:4]} {r['frac_mean'] * 100:5.1f}±{r['frac_sd'] * 100:4.1f}%{flag}"
            print(head)

        # --- pre-registered kill-rule on ARC (the headline anchor) ---
        B = ck_out.get("B_plainCE_indomain", {}).get("per_task", {}).get("arc_challenge")
        D = ck_out.get("D_plainCE_OOD", {}).get("per_task", {}).get("arc_challenge")
        if B and D:
            rb, rd = B["frac_mean"], D["frac_mean"]
            ratio = rd / rb if rb else float("nan")
            if not D["converged"]:
                verdict = ("DOMAIN EXPOSURE (strong): the OOD plain-CE arm does not even "
                           "converge -> generic FT off-domain recovers nothing")
            elif ratio >= 2 / 3:
                verdict = ("ANY FINE-TUNING: rec(D) >= 2/3 rec(B) -> domain is not what "
                           "matters; Limitations paragraph is replaced by this result")
            elif ratio <= 1 / 3:
                verdict = ("DOMAIN EXPOSURE: rec(D) <= 1/3 rec(B) -> re-scope 'generic "
                           "fine-tuning' to 'fine-tuning with domain exposure' paper-wide")
            else:
                verdict = ("PARTIAL: rec(D)/rec(B) between 1/3 and 2/3 -> report the "
                           "number, choose no reading")
            ck_out["arc_killrule"] = {"rec_B_indomain": rb, "rec_D_ood": rd,
                                      "ratio_D_over_B": ratio, "verdict": verdict}
            print(f"\n  ARC kill-rule: rec(B)={rb * 100:.1f}%  rec(D)={rd * 100:.1f}%  "
                  f"D/B={ratio:.2f}\n  -> {verdict}")

        out["checkpoints"][ck] = ck_out

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
