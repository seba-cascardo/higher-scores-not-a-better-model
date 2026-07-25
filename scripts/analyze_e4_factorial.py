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

The one kill-rule NOT applied verbatim is the third-axis rule in net_effect(): as
pre-registered it compared two point estimates with no variances, so noise alone
could fire it -- and it did (the two shifts differ by 0.07 with a standard error of
0.25). Corrected 2026-07-25 to require the DIFFERENCE BETWEEN THE SHIFTS to clear
2 sigma; the deviation and its reason are recorded in the ledger (B50).

  python scripts/analyze_e4_factorial.py
"""
import json
import os
import sys

import numpy as np

# The report prints non-ASCII (sigma, warning marks); a Windows console defaults to
# cp1252 and would die mid-report instead of writing the summary.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
                # The convergence gate lumps "did nothing" together with "made it
                # worse", and in E4 that distinction IS the finding (the OOD arms go
                # ~10pp BELOW base on TruthfulQA). Separate them explicitly.
                harmful = eff < 0
                status = ("harmful" if harmful.all() else
                          "no_op" if noop.all() else
                          "converged" if not noop.any() else "mixed")
                rec[t] = {
                    "arm_mean": float(arm.mean()), "arm_sd": float(arm.std(ddof=1)) if len(arm) > 1 else 0.0,
                    "effect_mean_pp": float(eff.mean() * 100),
                    "frac_mean": float(fr.mean()), "frac_sd": float(fr.std(ddof=1)) if len(fr) > 1 else 0.0,
                    "n_seeds": int(len(arm)),
                    "n_seeds_noop": int(noop.sum()),
                    "n_seeds_harmful": int(harmful.sum()),
                    "status": status,
                    "converged": bool((~noop).all()),
                }
            ck_out[name] = {"status": "ok" if not missing else "partial",
                            "n_seeds_found": len(got), "missing": missing, "per_task": rec}

            head = f"  {name:<24}"
            for t in TASKS:
                if t in rec:
                    r = rec[t]
                    flag = {"converged": "", "no_op": " [NO-OP]",
                            "harmful": " [HARMFUL]", "mixed": " [MIXED]"}[r["status"]]
                    head += f"  {t[:4]} {r['frac_mean'] * 100:5.1f}±{r['frac_sd'] * 100:4.1f}%{flag}"
            print(head)
            for t in TASKS:
                if t in rec and rec[t]["status"] == "harmful":
                    print(f"      ⚠ {t}: {rec[t]['effect_mean_pp']:+.1f}pp vs base "
                          f"({rec[t]['n_seeds_harmful']}/{rec[t]['n_seeds']} seeds below base) "
                          f"-- spillover damage, report it")

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

    out["net_effect"] = net_effect()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


# Published reference points of the off-axis spectrum (ledger B48). ratio =
# |cos(diff, v_inf)| / |cos(diff, random)|; 1.0 = indistinguishable from chance =
# maximally off-axis. These are NOT recomputed here -- they are cited, together
# with the seed dispersion that produced them, because the third-axis kill-rule
# below is a comparison between shifts and needs every term's variance.
# Sources, verified against the raw per-seed ratios:
#   1.85 +- 0.13  specs/2026-07-24-e2-seed-replication-results.md:60-63
#                 (ratios 1.74 / 1.81 / 2.00)
#   2.68 +- 0.34  specs/2026-07-24-e1-same-parameterization-control-results.md:88-91
#                 (ratios 2.92 / 2.82 / 2.29)
# Align-LoRA is a single published point with no replicates.
# The anchors are stored rounded as published; using 2.676667 instead of 2.68
# moves the shift difference from 0.28 to 0.29 sigma, which changes nothing.
SPECTRUM_REF = [
    ("contrastive in-domain (canonical, E2)", 1.85, 0.13, 3, "in", "contrastive"),
    ("same-param plain-CE in-domain (E1)", 2.68, 0.34, 3, "in", "plain-CE"),
    ("Align-LoRA (higher capacity)", 3.93, None, 1, "in", "plain SFT"),
]
CONTRASTIVE_IN, PLAINCE_IN = SPECTRUM_REF[0], SPECTRUM_REF[1]


def sem(sd, n):
    """Standard error of a mean; 0 for single published points with no replicates."""
    return float(sd) / np.sqrt(n) if (sd is not None and n > 1) else 0.0


def net_effect():
    """The two new spectrum points, and how much each axis moves the ratio."""
    cells = {"C_contrastive_OOD": "direct", "D_plainCE_OOD": "plain_ce"}
    got = {}
    print(f"\n{'=' * 74}\nnet-effect (off-axis ratio)\n{'=' * 74}")
    for name, loss in cells.items():
        rs, cs = [], []
        for s in (0, 1, 2):
            p = os.path.join(E4, f"neteffect_ood_{loss}_seed{s}.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                sm = json.load(f)["summary"]
            rs.append(float(sm["ratio"]))
            cs.append(float(sm["mean_abs_cos_vinf"]))
        if not rs:
            print(f"  {name:<22} MISSING")
            continue
        got[name] = {"ratio_mean": float(np.mean(rs)),
                     "ratio_sd": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
                     "abs_cos_vinf_mean": float(np.mean(cs)), "n_seeds": len(rs),
                     "ratios": rs}
        g = got[name]
        print(f"  {name:<22} ratio {g['ratio_mean']:.2f} ± {g['ratio_sd']:.2f}   "
              f"|cos| {g['abs_cos_vinf_mean']:.3f}   (n={g['n_seeds']})")

    if len(got) < 2:
        return got

    rows = SPECTRUM_REF + [
        ("contrastive OOD (E4-C, NEW)", got["C_contrastive_OOD"]["ratio_mean"],
         got["C_contrastive_OOD"]["ratio_sd"], got["C_contrastive_OOD"]["n_seeds"], "OOD", "contrastive"),
        ("plain-CE OOD (E4-D, NEW)", got["D_plainCE_OOD"]["ratio_mean"],
         got["D_plainCE_OOD"]["ratio_sd"], got["D_plainCE_OOD"]["n_seeds"], "OOD", "plain-CE"),
    ]
    print("\n  spectrum (1.0 = chance = maximally off-axis):")
    for n, r, sd, ns, dom, obj in sorted(rows, key=lambda x: x[1]):
        disp = f"± {sd:.2f}" if sd is not None and ns > 1 else "(single point)"
        print(f"    {r:.2f} {disp:<14} {n:<40} [{dom:>3}, {obj}]")

    c_in, c_in_sd, c_in_n = CONTRASTIVE_IN[1], CONTRASTIVE_IN[2], CONTRASTIVE_IN[3]
    p_in, p_in_sd, p_in_n = PLAINCE_IN[1], PLAINCE_IN[2], PLAINCE_IN[3]
    c_ood = got["C_contrastive_OOD"]["ratio_mean"]
    d_ood = got["D_plainCE_OOD"]["ratio_mean"]
    se_c_in, se_p_in = sem(c_in_sd, c_in_n), sem(p_in_sd, p_in_n)
    se_c_ood = sem(got["C_contrastive_OOD"]["ratio_sd"], got["C_contrastive_OOD"]["n_seeds"])
    se_d_ood = sem(got["D_plainCE_OOD"]["ratio_sd"], got["D_plainCE_OOD"]["n_seeds"])

    def shift(a, se_a, b, se_b):
        """b -> a, with the standard error of the difference and its z.

        sigma is None when the SE is zero (single-seed points with no dispersion):
        no evidence is not infinite evidence, and returning inf here would let the
        gate fire on a difference of 1e-4. None propagates to "cannot decide".
        """
        d, se = a - b, float(np.hypot(se_a, se_b))
        return {"shift": float(d), "se": se,
                "sigma": (float(abs(d) / se) if se > 0 else None)}

    axes = {
        "objective_shift_indomain": shift(p_in, se_p_in, c_in, se_c_in),
        "objective_shift_ood": shift(d_ood, se_d_ood, c_ood, se_c_ood),
        "domain_shift_contrastive": shift(c_ood, se_c_ood, c_in, se_c_in),
        "domain_shift_plainCE": shift(d_ood, se_d_ood, p_in, se_p_in),
    }
    for k, v in axes.items():
        sg = f"{v['sigma']:.1f}σ" if v["sigma"] is not None else "no dispersion"
        print(f"  {k:<28} {v['shift']:+.2f} ± {v['se']:.2f}   ({sg})")

    # --- pre-registered kill-rule, CORRECTED (2026-07-25) ---------------------
    # The original rule fired on abs(domain_shift) >= abs(objective_shift): two
    # point estimates compared with no variances, so noise alone could trigger it
    # (it did: the two shifts differ by 0.07). A third axis is only declared when
    # the DIFFERENCE BETWEEN THE SHIFTS clears 2 sigma.
    # Note the common term cancels: (c_ood - c_in) - (p_in - c_in) = c_ood - p_in,
    # so the contrastive in-domain anchor drops out and does not get counted twice.
    # That cancellation is exact, not approximate: Cov(S_dom, S_obj) = Var(c_in),
    # so Var(S_dom - S_obj) = Var(c_ood) + Var(p_in). Treating the shifts as
    # independent would add 2*Var(c_in) that is not there (SE 0.269 instead of
    # 0.247) -- i.e. it would UNDERSTATE the power of the test, not overstate it.
    #
    # Two honest caveats about the threshold itself, neither of which is reachable
    # from the published data but both of which matter if this is ever rerun:
    #   * z=2 over an sd estimated from 3 seeds is anti-conservative -- the
    #     two-sided 95% t critical value at 2 df is 4.30. The rule as
    #     pre-registered is in sigmas, so it stays, but a result at ~2.1 sigma
    #     would NOT be significant at 95% by t.
    #   * which axis "dominates" is a statement about MAGNITUDE, so the branch
    #     below compares |shift|, not signed shift. With shifts of opposite sign
    #     the signed comparison would report the wrong winner.
    SIGMA_GATE, T_CRIT_2DF = 2.0, 4.30
    diff = shift(c_ood, se_c_ood, p_in, se_p_in)
    sig = diff["sigma"]
    ordered = sig is not None and sig >= SIGMA_GATE
    dom_mag = abs(axes["domain_shift_contrastive"]["shift"])
    obj_mag = abs(axes["objective_shift_indomain"]["shift"])
    if sig is None:
        verdict = ("CANNOT DECIDE: the shift difference has no dispersion (single-seed "
                   "points) -> no evidence either way; do NOT read this as a third axis")
    elif not ordered:
        verdict = ("NOT ORDERABLE: domain is a real axis (domain shift "
                   f"{axes['domain_shift_contrastive']['shift']:+.2f} = "
                   f"{axes['domain_shift_contrastive']['sigma']:.1f}σ) but the two shifts "
                   f"differ by only {diff['shift']:+.2f} ± {diff['se']:.2f} "
                   f"({sig:.2f}σ < {SIGMA_GATE}σ) -> which axis weighs more is NOT decided "
                   "by these data; claim the domain axis, not the ordering")
    elif dom_mag > obj_mag:
        verdict = ("THIRD AXIS DOMINATES: domain moves the contrastive ratio more than the "
                   f"objective does, |{dom_mag:.2f}| vs |{obj_mag:.2f}|, difference "
                   f"{diff['shift']:+.2f} ± {diff['se']:.2f} ({sig:.1f}σ) -> off-axis "
                   "requires the CONJUNCTION objective x domain; E1's 'follows the "
                   "objective' must be qualified")
    else:
        verdict = ("OBJECTIVE DOMINATES: the objective shift exceeds the domain shift in "
                   f"magnitude, |{obj_mag:.2f}| vs |{dom_mag:.2f}| ({sig:.1f}σ) -> the "
                   "off-axis effect follows the training objective; domain is second-order")
    shown = f"{sig:.2f}σ" if sig is not None else "no dispersion"
    print(f"\n  shift difference (domain - objective): {diff['shift']:+.2f} ± {diff['se']:.2f} "
          f"({shown}, gate {SIGMA_GATE}σ; t crit at 2 df would be {T_CRIT_2DF})"
          f"\n  -> {verdict}")
    return {**got, "axis_shifts": axes, "shift_difference": diff,
            "sigma_gate": SIGMA_GATE, "t_critical_2df_for_reference": T_CRIT_2DF,
            # NOTE: this key means "the ordering between axes was resolved", which is
            # NOT what the pre-2026-07-25 `third_axis_triggered` meant (that one was
            # "domain >= objective"). It is True in the OBJECTIVE-DOMINATES branch too.
            # Renamed deliberately so no one compares old and new JSONs on one key.
            "axis_ordering_resolved": bool(ordered),
            "verdict": verdict,
            "spectrum": [[n, r, sd, ns, dom, obj] for n, r, sd, ns, dom, obj in rows]}


if __name__ == "__main__":
    main()
