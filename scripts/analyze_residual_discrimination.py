"""Residual discrimination AUC: after removing length + fluency from the cold-MC
score, is there ANY gold-vs-distractor signal left on this channel?

This closes the decisive open residual from the adversarial panel (2026-06-25):
is the correctness signal GENUINELY ABSENT on the cold-MC last-token channel
(strong "broken readout": residual AUC ~ 0.5) or PRESENT-BUT-SWAMPED by
fluency/length (weaker "miscalibration": residual AUC clearly > 0.5)?

Uses the self-consistent same-scaffold premise file (answer_only_<task>.json from
score_mc_premise_pod.py), which carries, under ONE scaffold and for both arms:
  conditional[arm][doc_id] = logP(option | Q)        (the cold-MC score)
  answer_only[arm][doc_id] = logP(option | "Answer:") (fluency/answer-prior proxy)
  charlen[doc_id], gold[doc_id]

Three per-option scores (acc_norm = divide each by char-length, matching the +35):
  naive   = cond_n                 (raw cold-MC ranking)
  PMI     = cond_n - answer_only_n (length already out via acc_norm; fluency/prior
                                    removed by subtracting the answer-only term)
  ans_only= answer_only_n          (pure fluency/prior, as a floor)

AUC(gold > distractor) is the Mann-Whitney probability that the gold outranks a
random distractor, averaged over all gold-distractor pairs in a subset (chance=0.5,
independent of #options). We report it for the WHOLE set and for the BASE-WRONG
subset (where the +35 lives). If PMI-AUC on base-wrong is ~0.5 -> the channel
carries no recoverable correctness signal once fluency is removed (the cold readout
is broken, not merely miscalibrated). If clearly >0.5 -> signal is present but
swamped, and the milder surface-form/miscalibration reading holds.

CPU-local, EUR0. Repo conventions: argparse, utf-8, progress prints, JSON save,
bootstrap CIs over items.
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


def pair_auc_items(scores_by_item: list[tuple[np.ndarray, int]]) -> tuple[float, int]:
    """AUC(gold > distractor) over all gold-distractor pairs pooled across items.
    Ties count 0.5. Returns (auc, n_pairs)."""
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


def boot_auc_ci(scores_by_item: list[tuple[np.ndarray, int]], n_boot: int, seed: int):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--premise-file", type=Path,
                    default=Path("runs/vinf_causal/answer_only_arc_challenge.json"))
    ap.add_argument("--arm", default="base", choices=["base", "mc"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pf = json.load(args.premise_file.open(encoding="utf-8"))
    task = pf.get("task", "?")
    print("=" * 78, flush=True)
    print(f"RESIDUAL DISCRIMINATION AUC  |  task={task}  arm={args.arm}  "
          f"premise={pf.get('neutral_premise')!r}", flush=True)
    print("=" * 78, flush=True)
    cond = pf["conditional"][args.arm]
    ao = pf["answer_only"][args.arm]
    gold_all, clen_all = pf["gold"], pf["charlen"]
    ids = sorted(set(int(k) for k in cond) & set(int(k) for k in ao)
                 & set(int(k) for k in gold_all) & set(int(k) for k in clen_all))
    print(f"  items: {len(ids)}   sanity (premise file): {pf.get('sanity')}", flush=True)

    naive_all, pmi_all, ans_all = [], [], []     # (scores, gold) per item, full set
    naive_bw, pmi_bw, ans_bw = [], [], []        # base-WRONG subset (naive argmax != gold)
    naive_br = []                                # base-RIGHT control (naive)
    pmi_br = []
    for did in ids:
        k = str(did)
        c = np.array(cond[k], dtype=np.float64)
        a = np.array(ao[k], dtype=np.float64)
        cl = np.array(clen_all[k], dtype=np.float64)
        gold = int(gold_all[k])
        if not (len(c) == len(a) == len(cl)) or gold >= len(c):
            continue
        cn = c / cl
        an = a / cl
        pmi = cn - an
        naive_all.append((cn, gold)); pmi_all.append((pmi, gold)); ans_all.append((an, gold))
        if int(np.argmax(cn)) != gold:          # base cold-MC WRONG (where the +35 lives)
            naive_bw.append((cn, gold)); pmi_bw.append((pmi, gold)); ans_bw.append((an, gold))
        else:
            naive_br.append((cn, gold)); pmi_br.append((pmi, gold))

    def report(label, sbi):
        auc, lo, hi = boot_auc_ci(sbi, args.n_boot, args.seed)
        print(f"      {label:34s} AUC={auc:.3f} CI[{lo:.3f},{hi:.3f}]  (n_items={len(sbi)})", flush=True)
        return dict(auc=auc, ci=[lo, hi], n=len(sbi))

    out = {}
    print(f"\n  --- WHOLE SET (n={len(naive_all)}) ---  [AUC(gold > distractor), chance=0.5]", flush=True)
    out["all_naive"] = report("naive cold-MC (cond_n)", naive_all)
    out["all_pmi"] = report("PMI residual (length+fluency out)", pmi_all)
    out["all_ans"] = report("answer-only (fluency floor)", ans_all)

    print(f"\n  --- BASE-WRONG subset (n={len(naive_bw)}) — where the +35 lives ---", flush=True)
    out["bw_naive"] = report("naive cold-MC (cond_n)", naive_bw)
    out["bw_pmi"] = report("PMI residual (length+fluency out)", pmi_bw)
    out["bw_ans"] = report("answer-only (fluency floor)", ans_bw)

    print(f"\n  --- BASE-RIGHT control (n={len(naive_br)}) ---", flush=True)
    out["br_naive"] = report("naive cold-MC (cond_n)", naive_br)
    out["br_pmi"] = report("PMI residual", pmi_br)

    # decisive read
    pmi_bw_auc = out["bw_pmi"]["auc"]
    pmi_bw_lo = out["bw_pmi"]["ci"][0]
    print(f"\n  === DECISIVE READ (base-wrong PMI residual AUC) ===", flush=True)
    if pmi_bw_lo > 0.55:
        verdict = (f"PRESENT-BUT-SWAMPED: residual AUC={pmi_bw_auc:.3f} (CI_lo {pmi_bw_lo:.3f} > 0.55) "
                   f">> chance -> the gold-tracking signal IS on the channel, merely out-ranked by "
                   f"fluency/length. The cold-MC failure is miscalibration/surface-form, not an empty "
                   f"readout. (Weaker claim for the paper.)")
    elif pmi_bw_auc <= 0.55:
        verdict = (f"ABSENT/BROKEN: residual AUC={pmi_bw_auc:.3f} ~ chance -> once length+fluency are "
                   f"removed, NO recoverable gold-vs-distractor signal remains on the cold-MC last-token "
                   f"channel. The readout is broken, not merely miscalibrated. (Strong eval-validity claim, "
                   f"consistent with W_know 6.4% and the off-axis-only fix.)")
    else:
        verdict = f"INTERMEDIATE: residual AUC={pmi_bw_auc:.3f} CI_lo={pmi_bw_lo:.3f} — weak/partial signal."
    print(f"      {verdict}", flush=True)
    out["verdict"] = verdict

    payload = dict(task=task, arm=args.arm, neutral_premise=pf.get("neutral_premise"), results=out)
    outp = args.out or (args.premise_file.parent / f"residual_discrimination_{task}_{args.arm}.json")
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {outp}", flush=True)


if __name__ == "__main__":
    main()
