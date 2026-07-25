"""FASE 1 -- does ordinary CE only know how to push the CORRECT option UP?

The hypothesis (fixed before looking at any number). Ordinary cross-entropy raises
log p(correct) but carries no term that pushes a distractor DOWN. HellaSwag rewards
exactly that -- the gold continuation is the plausible one, so lifting it is the
whole task -- and there the generic same-param control recovers everything
(Gemma 69.5%, Qwen 114.9%). TruthfulQA-mc1 demands the opposite: the gold answer
has to beat a semantically adjacent distractor, so lifting the gold drags its
neighbour up with it and the margin does not open. ARC sits in between. If true,
one mechanism explains the heterogeneity order in TWO families (HSwag > ARC > TQA),
the TQA damage in four independent arms, and why the contrastive objective is worth
anything at all.

Per item, per arm, against that arm's own base:
  d_correct    = logp(gold)_arm       - logp(gold)_base
  d_distractor = logp(distractor)_arm - logp(distractor)_base
  margin       = d_correct - d_distractor        (>0 separates, <0 collapses)

The distractor's IDENTITY is fixed by the BASE -- the non-gold option the base
ranked highest -- so both terms track the same option moving. Taking each arm's own
argmax would compare different options across arms and make the delta meaningless.
The arm's own top distractor is reported alongside, because that is what accuracy
actually sees.

Raw log-probs, not length-normalised: the claim is about what training does to
probabilities. ARC/HellaSwag are scored with acc_norm upstream, so accuracy here is
recomputed raw and will not match the published acc_norm -- that is expected and is
not the quantity under test.

Falsifiable predictions, fixed BEFORE running:
  P1  TQA, plain-CE arms:   d_distractor > 0, and margin ~ 0 or < 0
  P2  TQA, contrastive arm: margin clearly greater than the plain-CE arms
  P3  HSwag, plain-CE arms: margin positive (that is why they recover everything)

Death condition: if plain-CE does NOT raise the distractor's log-prob in TQA, the
theory dies and the TQA damage stays an open finding with no explanation. Declared
as such, and we move on.

Every cross-arm contrast is reported with its sigma. A difference under 2 sigma is
NOT claimed (the FASE 0 lesson).

  python scripts/analyze_tqa_distractor_mechanism.py
"""
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

E1 = "runs/e1_plain_ce"
E4 = "runs/e4_factorial"
E6 = "runs/e6_qwen_spine"
OUT = "runs/e4_factorial/tqa_distractor_mechanism.json"

# The contrast IS the test: TQA is where the generic control damages, HellaSwag is
# where it recovers >100%. ARC is the in-between anchor.
TASKS = ["truthfulqa_mc1", "hellaswag", "arc_challenge"]

# (label, family, objective, base path, [arm paths]). Checkpoints follow the specs:
# `best` everywhere except Qwen's plain-CE, where the best-val rule degenerated to
# step 1 (null offset) in 2/3 seeds, so `.final` is the honest arm.
ARMS = [
    ("Gemma contrastive canonical", "Gemma", "contrastive", f"{E1}/eval_base.json",
     [f"{E1}/eval_contrastive_canonical.json"]),
    ("Gemma plain-CE in-domain", "Gemma", "plain-CE", f"{E1}/eval_base.json",
     [f"{E1}/eval_lifts_seed{s}.json" for s in (0, 1, 2)]),
    ("Gemma contrastive OOD", "Gemma", "contrastive", f"{E1}/eval_base.json",
     [f"{E4}/eval_ood_direct_seed{s}.json" for s in (0, 1, 2)]),
    ("Gemma plain-CE OOD", "Gemma", "plain-CE", f"{E1}/eval_base.json",
     [f"{E4}/eval_ood_plain_ce_seed{s}.json" for s in (0, 1, 2)]),
    ("Qwen contrastive", "Qwen", "contrastive", f"{E6}/eval_base.json",
     [f"{E6}/eval_mc.json"]),
    ("Qwen plain-CE", "Qwen", "plain-CE", f"{E6}/eval_base.json",
     [f"{E6}/eval_plain_ce_seed{s}.final.json" for s in (0, 1, 2)]),
]


def load(path):
    """{task: {doc_hash: (logprobs array, gold index)}} from an lm-eval samples dump."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "samples" not in d:
        raise SystemExit(f"[ABORT] {path} has no per-sample dump (need --log_samples)")
    out = {}
    for t in TASKS:
        if t not in d["samples"]:
            continue
        per_doc = {}
        for s in d["samples"][t]:
            lp = np.array([float(r[0]) for r in s["filtered_resps"]])
            tgt = int(s["target"]) if not isinstance(s["target"], str) else None
            if tgt is None or not (0 <= tgt < len(lp)):
                continue
            per_doc[s["doc_hash"]] = (lp, tgt)
        if per_doc:
            out[t] = per_doc
    return out


def per_item(base_doc, arm_doc):
    """The three per-item quantities, over the docs both arms share."""
    keys = sorted(set(base_doc) & set(arm_doc))
    if not keys:
        return None
    d_cor, d_dis, d_dis_armtop, base_hit, arm_hit = [], [], [], [], []
    for k in keys:
        b_lp, tgt = base_doc[k]
        a_lp, tgt_a = arm_doc[k]
        if tgt != tgt_a or len(b_lp) != len(a_lp) or len(b_lp) < 2:
            continue
        others = [i for i in range(len(b_lp)) if i != tgt]
        # identity fixed by the BASE: the distractor the base ranked highest
        j = max(others, key=lambda i: b_lp[i])
        # what accuracy sees: the arm's own strongest distractor
        j_arm = max(others, key=lambda i: a_lp[i])
        d_cor.append(a_lp[tgt] - b_lp[tgt])
        d_dis.append(a_lp[j] - b_lp[j])
        d_dis_armtop.append(a_lp[j_arm] - b_lp[j_arm])
        base_hit.append(b_lp[tgt] > b_lp[j])
        arm_hit.append(a_lp[tgt] > max(a_lp[i] for i in others))
    if not d_cor:
        return None
    d_cor, d_dis = np.array(d_cor), np.array(d_dis)
    margin = d_cor - d_dis
    n = len(margin)

    def ms(x):
        return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x)))}

    return {
        "n_items": n,
        "d_correct": ms(d_cor),
        "d_distractor": ms(d_dis),
        "d_distractor_arm_top": ms(np.array(d_dis_armtop)),
        "margin": ms(margin),
        "frac_distractor_up": float((d_dis > 0).mean()),
        "frac_correct_up": float((d_cor > 0).mean()),
        "frac_margin_up": float((margin > 0).mean()),
        "acc_base_raw": float(np.mean(base_hit)),
        "acc_arm_raw": float(np.mean(arm_hit)),
    }


def agg(seeds, key):
    """Across seeds: mean and sd of a per-seed statistic (sd=0 for a single seed)."""
    v = np.array([s[key]["mean"] for s in seeds])
    return {"mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n_seeds": len(v),
            # for single-seed arms the item-level SE is the only dispersion available
            "item_se": float(np.mean([s[key]["se"] for s in seeds])),
            "per_seed": [float(x) for x in v]}


def sigma_between(a, b):
    """|a-b| in sigmas, using whichever dispersion each arm has (seeds if >1)."""
    sa = a["sd"] if a["n_seeds"] > 1 else a["item_se"]
    sb = b["sd"] if b["n_seeds"] > 1 else b["item_se"]
    s = float(np.hypot(sa, sb))
    d = a["mean"] - b["mean"]
    return {"diff": float(d), "se": s, "sigma": float(abs(d) / s) if s else float("inf")}


# Every "fraction of the lift" in the paper uses acc_norm for ARC/HellaSwag and acc
# for TQA/Winogrande (which define no length-normalised variant). This was NOT part
# of the pre-registered test -- it is a post-hoc check that ran because the margin
# analysis disagreed with the published HellaSwag ordering, and the margin is a raw
# log-prob quantity. Recomputing the inducibility fraction under both metrics says
# how much of that ordering is the normalisation.
METRIC_CELLS = {
    "Gemma": {"base": f"{E1}/eval_base.json", "contrastive": f"{E1}/eval_contrastive_canonical.json",
              "control": [f"{E1}/eval_lifts_seed{s}.json" for s in (0, 1, 2)]},
    "Qwen": {"base": f"{E6}/eval_base.json", "contrastive": f"{E6}/eval_mc.json",
             "control": [f"{E6}/eval_plain_ce_seed{s}.final.json" for s in (0, 1, 2)]},
}
METRIC_TASKS = {"arc_challenge": ["acc_norm,none", "acc,none"],
                "hellaswag": ["acc_norm,none", "acc,none"],
                "truthfulqa_mc1": ["acc,none"], "winogrande": ["acc,none"]}
MIN_LIFT_PP = 2.0  # below this the fraction's denominator is noise


def metric_sensitivity():
    """Inducibility fraction under the published metric and under raw acc."""
    def val(path, task, metric):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)["results"]
        return float(r[task][metric]) if task in r and metric in r[task] else None

    print(f"\n{'=' * 96}\nMETRIC SENSITIVITY (post-hoc, not pre-registered)\n{'=' * 96}")
    out = {}
    for fam, cell in METRIC_CELLS.items():
        print(f"\n  {fam}")
        print(f"    {'task':<16}{'metric':<10}{'lift pp':>9}{'control pp':>12}{'inducibility':>14}")
        out[fam] = {}
        for task, metrics in METRIC_TASKS.items():
            for m in metrics:
                b = val(cell["base"], task, m)
                c = val(cell["contrastive"], task, m)
                if b is None or c is None:
                    continue
                ctl = float(np.mean([val(p, task, m) for p in cell["control"]]))
                lift, eff = (c - b) * 100, (ctl - b) * 100
                readable = abs(lift) >= MIN_LIFT_PP
                frac = eff / lift if readable else None
                out[fam][f"{task}/{m}"] = {"base": b, "contrastive": c, "control_mean": ctl,
                                           "lift_pp": lift, "control_pp": eff,
                                           "inducibility": frac, "readable": readable}
                shown = f"{frac * 100:>13.1f}%" if readable else "   unreadable"
                print(f"    {task:<16}{m.replace(',none', ''):<10}{lift:>+9.2f}{eff:>+12.2f}{shown}"
                      f"{'' if readable else f'  (lift {lift:+.1f}pp < {MIN_LIFT_PP}pp)'}")
    return out


def main():
    results = {}
    print(f"{'=' * 96}\nFASE 1 -- distractor mechanism (raw log-probs, distractor fixed by base)\n{'=' * 96}")
    bases = {}
    for i, (label, fam, obj, base_path, arm_paths) in enumerate(ARMS, 1):
        print(f"\n[{i}/{len(ARMS)}] {label}  ({fam}, {obj}, {len(arm_paths)} seed(s))", flush=True)
        if base_path not in bases:
            if not os.path.exists(base_path):
                print(f"  [skip] base missing: {base_path}", flush=True)
                continue
            bases[base_path] = load(base_path)
        base = bases[base_path]

        per_task = {}
        for t in TASKS:
            seeds = []
            for p in arm_paths:
                if not os.path.exists(p):
                    print(f"  [skip] missing {p}", flush=True)
                    continue
                arm = load(p)
                if t not in arm or t not in base:
                    continue
                r = per_item(base[t], arm[t])
                if r:
                    seeds.append(r)
            if not seeds:
                continue
            per_task[t] = {k: agg(seeds, k) for k in
                           ("d_correct", "d_distractor", "d_distractor_arm_top", "margin")}
            per_task[t]["n_items"] = seeds[0]["n_items"]
            per_task[t]["n_seeds"] = len(seeds)
            for k in ("frac_distractor_up", "frac_correct_up", "frac_margin_up",
                      "acc_base_raw", "acc_arm_raw"):
                per_task[t][k] = float(np.mean([s[k] for s in seeds]))
            m, dc, dd = per_task[t]["margin"], per_task[t]["d_correct"], per_task[t]["d_distractor"]
            disp = f"±{m['sd']:.3f}" if m["n_seeds"] > 1 else f"±{m['item_se']:.3f}(item)"
            print(f"    {t:<16} n={per_task[t]['n_items']:<4} "
                  f"d_correct {dc['mean']:+7.3f}  d_distr {dd['mean']:+7.3f}  "
                  f"margin {m['mean']:+7.3f} {disp:<14} "
                  f"distr_up {per_task[t]['frac_distractor_up'] * 100:5.1f}%", flush=True)
        results[label] = {"family": fam, "objective": obj, "n_seeds": len(arm_paths),
                          "base": base_path, "arms": arm_paths, "per_task": per_task}

    # ---- the three pre-registered predictions -------------------------------
    print(f"\n{'=' * 96}\nPRE-REGISTERED PREDICTIONS\n{'=' * 96}")
    verdicts = {}

    def get(label, task, key):
        return results.get(label, {}).get("per_task", {}).get(task, {}).get(key)

    plain_tqa = [l for l in results if results[l]["objective"] == "plain-CE"
                 and get(l, "truthfulqa_mc1", "margin")]
    p1_rows = []
    for l in plain_tqa:
        dd, m = get(l, "truthfulqa_mc1", "d_distractor"), get(l, "truthfulqa_mc1", "margin")
        up = dd["mean"] > 0
        p1_rows.append({"arm": l, "d_distractor": dd["mean"], "margin": m["mean"],
                        "distractor_raised": bool(up)})
        print(f"  P1  {l:<28} d_distr {dd['mean']:+7.3f} ({'UP' if up else 'DOWN'})  "
              f"margin {m['mean']:+7.3f}")
    p1 = bool(p1_rows) and all(r["distractor_raised"] for r in p1_rows)
    verdicts["P1_plainCE_raises_distractor_in_TQA"] = {"holds": p1, "rows": p1_rows}
    print(f"  -> P1 {'HOLDS' if p1 else 'FAILS'}: plain-CE "
          f"{'raises' if p1 else 'does NOT raise'} the distractor in TQA"
          f"{'' if p1 else '  ** DEATH CONDITION: the theory dies **'}")

    p2 = {}
    for fam in ("Gemma", "Qwen"):
        con = [l for l in results if results[l]["family"] == fam
               and results[l]["objective"] == "contrastive" and get(l, "truthfulqa_mc1", "margin")]
        pla = [l for l in results if results[l]["family"] == fam
               and results[l]["objective"] == "plain-CE" and get(l, "truthfulqa_mc1", "margin")]
        for c in con:
            for p in pla:
                s = sigma_between(get(c, "truthfulqa_mc1", "margin"), get(p, "truthfulqa_mc1", "margin"))
                p2[f"{c} vs {p}"] = s
                mark = "CLAIMED" if s["sigma"] >= 2 and s["diff"] > 0 else "not claimed (<2σ or wrong sign)"
                print(f"  P2  {c:<28} vs {p:<26} Δmargin {s['diff']:+7.3f} ± {s['se']:.3f} "
                      f"({s['sigma']:.1f}σ) {mark}")
    verdicts["P2_contrastive_margin_beats_plainCE_in_TQA"] = p2

    p3_rows = []
    for l in [l for l in results if results[l]["objective"] == "plain-CE"
              and get(l, "hellaswag", "margin")]:
        m = get(l, "hellaswag", "margin")
        disp = m["sd"] if m["n_seeds"] > 1 else m["item_se"]
        sig = abs(m["mean"]) / disp if disp else float("inf")
        pos = m["mean"] > 0 and sig >= 2
        p3_rows.append({"arm": l, "margin": m["mean"], "sigma": sig, "positive": bool(pos)})
        print(f"  P3  {l:<28} HSwag margin {m['mean']:+7.3f} ({sig:.1f}σ) "
              f"{'POSITIVE' if pos else 'not positive at 2σ'}")
    verdicts["P3_plainCE_margin_positive_in_HSwag"] = {"rows": p3_rows,
                                                       "holds": bool(p3_rows) and all(r["positive"] for r in p3_rows)}

    out = {"predictions_fixed_before_running": {
        "P1": "TQA, plain-CE arms: d_distractor > 0 and margin ~0 or <0",
        "P2": "TQA, contrastive arm: margin clearly greater than plain-CE",
        "P3": "HSwag, plain-CE arms: margin positive"},
        "distractor_identity": "fixed by the BASE (highest-ranked non-gold option)",
        "metric": "raw log-prob (not length-normalised); acc_* here are raw, not acc_norm",
        "arms": results, "verdicts": verdicts,
        "metric_sensitivity": metric_sensitivity()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
