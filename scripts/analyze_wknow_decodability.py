"""Is W_know ≈ chance REAL (gold not decodable options-absent) or PROBE-UNDERPOWERED? €0 local.

ClaraVT decision 2026-06-24: the faithful-KAPPA W_know hit ≈chance at the options-ABSENT
prompt-final. This script firms WHY, deciding the KAPPA section of the paper (option B —
KAPPA's precondition structurally fails in the off-axis channel) WITHOUT spending pod money.

THE TEST: sweep regularization C for BOTH k-class probes on the same residual:
  - W_know (target = GOLD option): the knowledge probe KAPPA needs.
  - W_pred (target = GREEDY option): POSITIVE CONTROL — the model's own behavior, a function
    of the residual, so it SHOULD be decodable.
Same probe, same data, same dim — the only difference is the target.
  * If W_pred decodes (>chance) but W_know stays ≈chance across ALL C  -> the gold INDEX is
    genuinely NOT linearly present in the options-absent prompt-final residual (the model can't
    know which arbitrary index is correct without seeing the options). NOT underpowering.
    => KAPPA's knowledge subspace cannot be instantiated in this channel. Option B firmed.
  * If W_know climbs with the right C -> it WAS underpowered; revisit.

Contrast (cited, not recomputed here): the project's post-gen correctness probe is AUC 0.94-0.96
(runs/correctness_probe_postgen). So the knowledge IS decodable — post-generation / mid-layer,
NOT static-linear at the options-absent prompt-final. Channel/position-specific, not absent.

Usage (LOCAL): python scripts/analyze_wknow_decodability.py \
    --resid runs/gate2/promptfinal_residual_arc.pt --out runs/gate2/wknow_decodability.json
Compile-check: python -c "import ast;ast.parse(open('scripts/analyze_wknow_decodability.py').read())"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.linear_model import LogisticRegression

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def split_by_gold(doc_ids, golds, seed=0, frac_tr=0.7):
    rng = np.random.default_rng(seed)
    tr = set()
    g = np.asarray(golds)
    for cls in sorted(set(g.tolist())):
        ids = [d for d, gg in zip(doc_ids, golds) if gg == cls]
        ids = list(rng.permutation(ids))
        tr.update(ids[:int(round(frac_tr * len(ids)))])
    return tr


def dev_acc(X, y, tr_mask, C):
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X[tr_mask], y[tr_mask])
    return float((clf.predict(X[~tr_mask]) == y[~tr_mask]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resid", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    blob = torch.load(args.resid, map_location="cpu", weights_only=False)
    layers = [int(x) for x in blob["layers"]]
    data = blob["data"]
    kc = Counter(int(data[d]["n_opt"]) for d in data)
    dom_k = kc.most_common(1)[0][0]
    doc_ids = sorted(d for d in data if int(data[d]["n_opt"]) == dom_k)
    golds = [int(data[d]["gold"]) for d in doc_ids]
    greedys = [int(data[d]["greedy"]) for d in doc_ids]
    chance = 1.0 / dom_k
    print(f"  {len(doc_ids)} items (dominant k={dom_k}, chance={chance:.3f}); layers={layers}", flush=True)

    tr_set = split_by_gold(doc_ids, golds, seed=args.seed)
    tr_mask = np.array([d in tr_set for d in doc_ids])
    y_gold = np.array(golds)
    y_greedy = np.array(greedys)

    per_layer = {}
    best_know = 0.0
    best_pred = 0.0
    for lpos, li in enumerate(layers):
        X = np.stack([data[d]["residual"][lpos].numpy() for d in doc_ids]).astype(np.float64)
        row = {"C_grid": C_GRID, "know": [], "pred": []}
        for C in C_GRID:
            ak = dev_acc(X, y_gold, tr_mask, C)
            ap_ = dev_acc(X, y_greedy, tr_mask, C)
            row["know"].append(ak)
            row["pred"].append(ap_)
            best_know = max(best_know, ak)
            best_pred = max(best_pred, ap_)
        per_layer[str(li)] = row
        print(f"  L{li}: know(maxC)={max(row['know']):.3f}  pred(maxC)={max(row['pred']):.3f}  "
              f"(C-grid know={[round(x,2) for x in row['know']]})", flush=True)

    # verdict: gold not decodable (≈chance across all C) while greedy is -> options-absent, not underpowered
    know_at_chance = best_know < chance + 0.10            # best gold dev-acc within 10pp of chance
    pred_decodes = best_pred > chance + 0.10              # greedy clearly above chance
    if know_at_chance and pred_decodes:
        verdict = ("REAL: gold-index NOT linearly decodable from the options-absent prompt-final "
                   "(W_know ≈ chance across ALL C) while greedy IS (positive control decodes). "
                   "NOT underpowering -> KAPPA's knowledge subspace cannot be instantiated in this "
                   "channel. Option B firmed.")
    elif not know_at_chance:
        verdict = ("REVISIT: W_know climbs above chance with regularization -> it was partly "
                   "underpowered. Re-examine before claiming non-decodability.")
    else:
        verdict = ("INCONCLUSIVE: greedy control also near chance -> probe/data issue, not a clean "
                   "channel finding. Inspect.")

    payload = {
        "resid_source": str(args.resid), "dominant_k": dom_k, "chance": chance,
        "n_items": len(doc_ids), "layers": layers, "C_grid": C_GRID,
        "best_dev_acc_know_gold": best_know, "best_dev_acc_pred_greedy": best_pred,
        "per_layer": per_layer,
        "postgen_correctness_probe_AUC_cited": "0.94-0.96 (runs/correctness_probe_postgen; "
            "knowledge IS decodable post-gen/mid-layer, NOT static-linear at options-absent prompt-final)",
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  best dev_acc: W_know(gold)={best_know:.3f}  W_pred(greedy)={best_pred:.3f}  chance={chance:.3f}",
          flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
