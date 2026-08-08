"""P4 — Power of the present-but-swamped BASE arm.

B7b left the honest caveat: the "signal is present-but-swamped" claim clears its
CI_lo (>0.55) only on the ADAPTER arm (AUC 0.667, n=73); on the BASE arm the
PMI-residual AUC is 0.574 with CI_lo 0.520 — INTERMEDIATE, does NOT clear 0.55.
The plan (P4) says: raise power with the pre-specified pool "base-wrong (n=204) +
S_cot_coldwrong (n=191), dedup by doc_id" and re-check the gate CI_lo > 0.55.

This script re-computes the base-arm PMI-residual AUC(gold>distractor) over several
DEFENSIBLE pool definitions so the number — not the wording — decides the gate:

  P1 base_wrong_premise   (n=204)  SANITY: must reproduce B7b base = 0.574/CI[.520,.627]
  P2 s_cot_coldwrong      (n=191)  the ADAPTER-INDEPENDENT "the base KNOWS it (CoT-right)
                                   but cold-MC fails" pool — cleanest signal pool
  P3 bw_premise ∧ CoT-right         the premise-consistent "knows" pool (no cross-harness
                                   membership leak)
  P4 union dedup          (n=228)  the LITERAL spec: base-wrong(premise) ∪ S_cot_coldwrong

All pools are SCORED by the same premise-file PMI residual (cond_n - answer_only_n),
matching B7b exactly; only the item MEMBERSHIP differs. Caveat surfaced in output:
P4's +24 items are base-RIGHT under premise-scoring, so they bias the AUC upward
(pro-gate) — reported explicitly, not hidden.

Gate (pre-registered, B7b): CI_lo > 0.55 on the pre-specified pool -> "present" is
affirmable on the base arm and the claim rises intermediate -> present.
CI_lo <= 0.55 -> the current text ("intermediate, claim rests on the adapter arm")
is already correct — freeze it and close the pending.

CPU-local, EUR0. Bootstrap over items, seed=42, n_boot=2000 (matches
analyze_residual_discrimination.py so P1 reproduces bit-for-bit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def pair_auc_items(scores_by_item):
    """AUC(gold > distractor) over all gold-distractor pairs pooled across items.
    Ties count 0.5. (Identical to analyze_residual_discrimination.py.)"""
    wins = 0.0
    npairs = 0
    for s, gold in scores_by_item:
        g = s[gold]
        for j in range(len(s)):
            if j == gold:
                continue
            npairs += 1
            if g > s[j]:
                wins += 1.0
            elif g == s[j]:
                wins += 0.5
    return (wins / npairs if npairs else float("nan")), npairs


def boot_auc_ci(scores_by_item, n_boot, seed):
    """Bootstrap CI by resampling ITEMS (preserves within-item pairing)."""
    n = len(scores_by_item)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, _ = pair_auc_items([scores_by_item[i] for i in idx])
        aucs.append(a)
    point, _ = pair_auc_items(scores_by_item)
    return point, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def opt_texts(doc):
    ch = doc.get("choices")
    if isinstance(ch, dict) and "text" in ch:
        return list(ch["text"])
    if isinstance(ch, list) and ch and isinstance(ch[0], str):
        return list(ch)
    mc1 = doc.get("mc1_targets")
    if isinstance(mc1, dict) and "choices" in mc1:
        return list(mc1["choices"])
    raise KeyError(f"no options in doc (keys={list(doc.keys())})")


def load_samples(path, task):
    d = json.load(path.open(encoding="utf-8"))
    return {int(s["doc_id"]): s for s in d["samples"][task]}


def cond_lp(s):
    return np.array([float(r[0]) for r in s["filtered_resps"]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("runs/vinf_causal"))
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    D = args.dir
    print("=" * 78, flush=True)
    print(f"P4 BASE-ARM POWER  |  task={args.task}  gate=CI_lo>{args.gate}", flush=True)
    print("=" * 78, flush=True)

    # ---- premise file: per-item PMI residual scores (the B7b channel) -------------
    pf = json.load((D / f"answer_only_{args.task}.json").open(encoding="utf-8"))
    cond = pf["conditional"]["base"]
    ao = pf["answer_only"]["base"]
    gold_all, clen_all = pf["gold"], pf["charlen"]
    prem_ids = sorted(set(int(k) for k in cond) & set(int(k) for k in ao)
                      & set(int(k) for k in gold_all) & set(int(k) for k in clen_all))
    print(f"  [premise] {(D / f'answer_only_{args.task}.json').name}: {len(prem_ids)} items "
          f"sanity={pf.get('sanity')}", flush=True)

    pmi_by_id = {}       # did -> (pmi_scores, gold)
    naive_by_id = {}
    ans_by_id = {}
    bw_premise = set()   # base-wrong under premise scoring (acc_norm)
    for did in prem_ids:
        k = str(did)
        c = np.array(cond[k], dtype=np.float64)
        a = np.array(ao[k], dtype=np.float64)
        cl = np.array(clen_all[k], dtype=np.float64)
        gold = int(gold_all[k])
        if not (len(c) == len(a) == len(cl)) or gold >= len(c):
            continue
        cn = c / cl
        an = a / cl
        pmi_by_id[did] = (cn - an, gold)
        naive_by_id[did] = (cn, gold)
        ans_by_id[did] = (an, gold)
        if int(np.argmax(cn)) != gold:
            bw_premise.add(did)
    print(f"  [premise] base-wrong (argmax cond_n != gold): n={len(bw_premise)}", flush=True)

    # ---- lm-eval d_null + arbiter: reproduce S_cot_coldwrong membership -----------
    null = load_samples(D / "d_null.json", args.task)
    arb = {}
    # Prefer the fixed-parser v2 arbiter when present (same rule as
    # analyze_arbiter_arc_cot.py / analyze_cold_mc_calibration.py). Reading the
    # pre-fix v1 drops doc 128 — base-wrong, unparsed under the old letter-parser
    # — from S_cot_coldwrong, giving 191 members instead of 192.
    _v2 = D / "arbiter_arc_cot_v2.json"
    a = json.load((_v2 if _v2.exists() else D / "arbiter_arc_cot.json").open(encoding="utf-8"))
    for r in a.get("results", []):
        arb[int(r["doc_id"])] = r
    bw_lmeval = set()
    s_cot_coldwrong = set()
    for did, s in null.items():
        texts = opt_texts(s["doc"])
        gold = int(s["target"])
        cl = np.array([max(1, len(t)) for t in texts], dtype=np.float64)
        b = cond_lp(s)
        if len(b) != len(cl) or gold >= len(b):
            continue
        b_wrong = int(np.argmax(b / cl)) != gold
        if b_wrong:
            bw_lmeval.add(did)
        if b_wrong and arb.get(did, {}).get("base_cot_correct") == 1:
            s_cot_coldwrong.add(did)
    cot_right = {did for did, r in arb.items() if r.get("base_cot_correct") == 1}
    print(f"  [lm-eval] base-wrong={len(bw_lmeval)}  S_cot_coldwrong(cold-wrong∧CoT-right)="
          f"{len(s_cot_coldwrong)}", flush=True)
    extra = (bw_premise | s_cot_coldwrong) - bw_premise
    print(f"  [union] base-wrong(premise) ∪ S_cot_coldwrong adds {len(extra)} items; of those, "
          f"base-RIGHT under premise-scoring (pro-gate bias): "
          f"{len(extra - bw_premise)} — all {len(extra)} are (they're premise-right by defn)",
          flush=True)

    # ---- pools (all scored by premise PMI) ---------------------------------------
    pools = {
        "P1_base_wrong_premise": bw_premise,
        "P2_s_cot_coldwrong": s_cot_coldwrong,
        "P3_bw_premise_and_CoTright": bw_premise & cot_right,
        "P4_union_dedup": bw_premise | s_cot_coldwrong,
    }
    pool_desc = {
        "P1_base_wrong_premise": "SANITY vs B7b base (0.574 / CI[.520,.627])",
        "P2_s_cot_coldwrong": "adapter-indep 'base KNOWS (CoT) but cold-MC fails' — cleanest",
        "P3_bw_premise_and_CoTright": "premise-consistent 'knows' pool (no cross-harness leak)",
        "P4_union_dedup": "LITERAL spec base-wrong(prem) ∪ S_cot_coldwrong (+24 premise-right, pro-gate)",
    }

    def scores_for(pool, by_id):
        # preserve sorted(prem_ids) order so P1 bootstrap reproduces the original exactly
        return [by_id[did] for did in prem_ids if did in pool and did in by_id]

    def report(name, pool):
        pmi_sbi = scores_for(pool, pmi_by_id)
        nav_sbi = scores_for(pool, naive_by_id)
        ans_sbi = scores_for(pool, ans_by_id)
        auc, lo, hi = boot_auc_ci(pmi_sbi, args.n_boot, args.seed)
        nav = boot_auc_ci(nav_sbi, args.n_boot, args.seed)
        an_ = boot_auc_ci(ans_sbi, args.n_boot, args.seed)
        passed = lo > args.gate
        print(f"\n  --- {name}  (n={len(pmi_sbi)}) ---  {pool_desc[name]}", flush=True)
        print(f"      PMI residual   AUC={auc:.3f} CI[{lo:.3f},{hi:.3f}]  "
              f"{'PASS (CI_lo>%.2f)' % args.gate if passed else 'fail (CI_lo<=%.2f)' % args.gate}",
              flush=True)
        print(f"      naive cold-MC  AUC={nav[0]:.3f} CI[{nav[1]:.3f},{nav[2]:.3f}]", flush=True)
        print(f"      answer-only    AUC={an_[0]:.3f} CI[{an_[1]:.3f},{an_[2]:.3f}]", flush=True)
        return dict(n=len(pmi_sbi),
                    pmi=dict(auc=auc, ci=[lo, hi], pass_gate=bool(passed)),
                    naive=dict(auc=nav[0], ci=[nav[1], nav[2]]),
                    ans_only=dict(auc=an_[0], ci=[an_[1], an_[2]]))

    out = {name: report(name, pool) for name, pool in pools.items()}

    # ---- verdict -----------------------------------------------------------------
    gate_pools = ["P2_s_cot_coldwrong", "P4_union_dedup"]  # the two spec-relevant ones
    passes = {p: out[p]["pmi"]["pass_gate"] for p in gate_pools}
    print("\n  === GATE READ (pre-registered CI_lo > %.2f) ===" % args.gate, flush=True)
    for p in gate_pools:
        print(f"      {p:26s} CI_lo={out[p]['pmi']['ci'][0]:.3f}  "
              f"-> {'PASS' if passes[p] else 'FAIL'}", flush=True)
    if all(passes.values()):
        verdict = ("PRESENT (base arm): every spec-relevant pool clears CI_lo>0.55 -> the "
                   "gold-tracking signal is present-but-swamped on the BASE arm too; draft 5.1 "
                   "rises intermediate->present.")
    elif not any(passes.values()):
        verdict = ("INTERMEDIATE FROZEN (base arm): no spec-relevant pool clears CI_lo>0.55 -> "
                   "the current draft text ('intermediate, claim rests on the adapter arm') is "
                   "already correct; freeze it and close the pending. Power does not rescue it.")
    else:
        verdict = ("SPLIT: pools disagree on the gate (%s). User decision on which pool ships — "
                   "note P4 has the +24 premise-right pro-gate bias; P2 is the clean pool."
                   % ", ".join(f"{p}={'PASS' if v else 'FAIL'}" for p, v in passes.items()))
    print(f"\n  VERDICT: {verdict}", flush=True)
    out["verdict"] = verdict
    out["meta"] = dict(task=args.task, gate=args.gate, n_boot=args.n_boot, seed=args.seed,
                       set_sizes=dict(bw_premise=len(bw_premise), bw_lmeval=len(bw_lmeval),
                                      s_cot_coldwrong=len(s_cot_coldwrong),
                                      union=len(bw_premise | s_cot_coldwrong)))

    outp = args.out or (D / f"p4_base_arm_power_{args.task}.json")
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  saved: {outp}", flush=True)


if __name__ == "__main__":
    main()
