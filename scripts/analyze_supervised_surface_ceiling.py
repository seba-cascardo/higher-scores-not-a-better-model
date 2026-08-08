"""Supervised surface ceiling: the STRONGEST interpretable re-scoring of the collapsed
cold-MC output scalars, to answer the critic's objection that the 2-dof oracle stack
(tuned for global acc, base-wrong AUC 0.502 < plain PMI 0.574) under-fits the baseline.

Fits held-out-by-item classifiers (logistic + non-linear GBM) to discriminate gold vs
distractor from per-option features derived ONLY from the output scalars (cond, answer-
only, char-length and their within-item transforms). Reports out-of-fold AUC(gold >
distractor) on the BASE-WRONG subset (where the +35 lives), vs the adapter's 0.830.

If even this supervised, AUC-objective, non-linear surface model caps near ~0.57, the
surface ceiling is FUNDAMENTAL for the collapsed-scalar channel (not a tuning artifact),
and the lift genuinely does not reduce to interpretable re-scoring. If it climbs toward
0.83, the critic is right and the earlier "19% recovery" overstated how broken the
readout is.

Scope note: this is still OUTPUT-SCALAR space, the fair surface
ceiling. The decisive activation-space probe (per-head o_proj-input, same cold-MC
distribution) is POD work — the activations are NOT in the local logprob JSONs.

CPU-local, EUR0. sklearn. Repo conventions: argparse, utf-8, progress prints, JSON save.
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def pair_auc(scores_by_item):
    wins, npairs = 0.0, 0
    for s, gold in scores_by_item:
        g = s[gold]
        for j in range(len(s)):
            if j == gold:
                continue
            npairs += 1
            wins += 1.0 if g > s[j] else (0.5 if g == s[j] else 0.0)
    return (wins / npairs) if npairs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--premise-file", type=Path,
                    default=Path("runs/vinf_causal/answer_only_arc_challenge.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pf = json.load(args.premise_file.open(encoding="utf-8"))
    task = pf.get("task", "?")
    print("=" * 80, flush=True)
    print(f"SUPERVISED SURFACE CEILING  |  task={task}  (held-out AUC-objective re-scoring)", flush=True)
    print("=" * 80, flush=True)
    cond, ao = pf["conditional"]["base"], pf["answer_only"]["base"]
    gold_all, clen_all = pf["gold"], pf["charlen"]
    ids = sorted(set(int(k) for k in cond) & set(int(k) for k in ao)
                 & set(int(k) for k in gold_all))

    # build per-item feature blocks. features vary WITHIN an item (so they can re-rank).
    items = []   # (X[opt,feat], gold, base_wrong_flag)
    feat_names = ["cond_n", "ao_n", "charlen", "pmi", "cond_centered", "ao_centered",
                  "cond_rank", "charlen_centered"]
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
        cond_rank = np.argsort(np.argsort(cn)).astype(np.float64)  # 0..n-1
        X = np.column_stack([cn, an, cl, pmi,
                             cn - cn.mean(), an - an.mean(),
                             cond_rank, cl - cl.mean()])
        base_wrong = int(np.argmax(cn)) != gold
        items.append((X, gold, base_wrong))
    n_items = len(items)
    print(f"  items: {n_items}   features: {feat_names}", flush=True)

    # out-of-fold predictions, held-out BY ITEM
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    idx_all = np.arange(n_items)
    oof_log = [None] * n_items
    oof_gbm = [None] * n_items
    for fold, (tr, te) in enumerate(kf.split(idx_all)):
        Xtr = np.vstack([items[i][0] for i in tr])
        ytr = np.concatenate([(np.arange(items[i][0].shape[0]) == items[i][1]).astype(int) for i in tr])
        sc = StandardScaler().fit(Xtr)
        log = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
        gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         random_state=args.seed).fit(Xtr, ytr)
        for i in te:
            Xi = items[i][0]
            oof_log[i] = log.predict_proba(sc.transform(Xi))[:, 1]
            oof_gbm[i] = gbm.predict_proba(Xi)[:, 1]
        print(f"    fold {fold + 1}/{args.folds} done (train {len(tr)} items, test {len(te)})", flush=True)

    # AUC(gold>distractor): whole set and base-wrong, for each model + PMI reference
    def auc_subset(scorer, mask):
        sbi = []
        for i in range(n_items):
            if mask(items[i]):
                sbi.append((scorer(i), items[i][1]))
        return pair_auc(sbi), len(sbi)

    pmi_scorer = lambda i: items[i][0][:, 3]      # pmi column
    cond_scorer = lambda i: items[i][0][:, 0]     # cond_n column
    log_scorer = lambda i: oof_log[i]
    gbm_scorer = lambda i: oof_gbm[i]

    print(f"\n  --- AUC(gold > distractor), chance=0.5 ---", flush=True)
    all_mask = lambda it: True
    bw_mask = lambda it: it[2]
    for name, sc in [("base cond_n (naive)", cond_scorer), ("PMI (subtract answer-only)", pmi_scorer),
                     ("supervised LOGISTIC (OOF)", log_scorer), ("non-linear GBM (OOF)", gbm_scorer)]:
        a_all, n_all = auc_subset(sc, all_mask)
        a_bw, n_bw = auc_subset(sc, bw_mask)
        print(f"      {name:30s}  whole={a_all:.3f} (n={n_all})   base-wrong={a_bw:.3f} (n={n_bw})",
              flush=True)

    bw_log, _ = auc_subset(log_scorer, bw_mask)
    bw_gbm, _ = auc_subset(gbm_scorer, bw_mask)
    ceiling = max(bw_log, bw_gbm, auc_subset(pmi_scorer, bw_mask)[0])
    print(f"\n  === VERDICT (vs adapter base-wrong AUC 0.830) ===", flush=True)
    if ceiling < 0.65:
        verdict = (f"SURFACE CEILING IS FUNDAMENTAL: the strongest supervised, held-out, AUC-objective "
                   f"surface re-scoring caps at base-wrong AUC {ceiling:.3f} (logistic {bw_log:.3f}, "
                   f"GBM {bw_gbm:.3f}) << adapter 0.830. The critic's 'under-fit baseline' objection is "
                   f"answered: no re-scoring of the collapsed scalars recovers the lift; the signal is "
                   f"upstream of them. (Decisive activation-space probe = POD.)")
    else:
        verdict = (f"BASELINE WAS UNDER-FIT: supervised surface re-scoring reaches base-wrong AUC "
                   f"{ceiling:.3f} -> the earlier oracle '19%/0.502' overstated the ceiling; the gap to "
                   f"the adapter (0.830) is smaller than claimed.")
    print(f"      {verdict}", flush=True)

    payload = dict(task=task, features=feat_names, folds=args.folds,
                   auc_base_wrong=dict(logistic=bw_log, gbm=bw_gbm,
                                       pmi=auc_subset(pmi_scorer, bw_mask)[0],
                                       cond_naive=auc_subset(cond_scorer, bw_mask)[0],
                                       adapter_reference=0.830),
                   ceiling=ceiling, verdict=verdict)
    outp = args.out or (args.premise_file.parent / f"supervised_surface_ceiling_{task}.json")
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {outp}", flush=True)


if __name__ == "__main__":
    main()
