"""P1 — activation-probe in-space: does a same-space linear reader recover the +35?

The prose-freeze gate of §5 (handoff 2026-06-26 §1d). Closes the information-set
confound of §5.2b: the adapter mutates ACTIVATIONS, the surface stack re-weights
collapsed SCALARS (AUC cap 0.574). This fits a linear probe on the SAME per-head
o_proj-input activations the adapter writes to — base, NO adapter, held-out by-item —
and reports the SAME pair_auc(gold>distractor) on the SAME base-wrong subset, so the
number is apples-to-apples with the adapter's 0.830 (analyze_offaxis_vs_interpretable_stack).

  ~0.83  -> signal present, any same-space reader extracts it -> PURE MISCALIBRATION
            (readout-claim strongest: lossy readout, signal intact upstream).
  ~0.52  -> the offset adds/rotates signal a linear same-space probe cannot read
            -> representational EDIT (stronger-and-different claim).
  W_know (LDA, 6.4% recovery) is the on-axis LOWER bound; this is the in-space UPPER bound.

KILL-RULE (forward-filter): the comparison is only valid same-distribution (cold-MC,
guaranteed by the extract reconstructing d_null arguments) AND same-space (per-head
o_proj-input adapter cells, guaranteed by the extract subset). The probe MUST be
held-out by-item (GroupKFold on item) — an in-sample fit on 12k dims is meaningless.

Input : runs/vinf_causal/coldmc_perhead_<task>.pt  (from extract_coldmc_perhead_pod.py)
Anchor: runs/vinf_causal/offaxis_vs_stack_<task>.json (adapter 0.830 / surface 0.574)
CPU-local, EUR0. sklearn (LogisticRegression + GroupKFold + StandardScaler); numpy fallback.

Usage:
  python scripts/analyze_activation_probe_inspace.py \
      --acts runs/vinf_causal/coldmc_perhead_arc_challenge.pt
Compile-check: python -c "import ast;ast.parse(open('scripts/analyze_activation_probe_inspace.py').read())"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # torch only needed to load the .pt
    torch = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def pair_auc(scores_by_item):
    """Identical metric to analyze_offaxis_vs_interpretable_stack.pair_auc."""
    wins, npairs = 0.0, 0
    for s, gold in scores_by_item:
        g = s[gold]
        for j in range(len(s)):
            if j == gold:
                continue
            npairs += 1
            wins += 1.0 if g > s[j] else (0.5 if g == s[j] else 0.0)
    return (wins / npairs) if npairs else float("nan")


def boot_pair_auc_ci(scores_by_item, n_boot, seed):
    """Bootstrap 95% CI by resampling ITEMS (not pairs)."""
    rng = np.random.default_rng(seed)
    arr = list(scores_by_item)
    n = len(arr)
    if n == 0:
        return (float("nan"),) * 2
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(pair_auc([arr[i] for i in idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_heldout_scores(X, y, groups, Cs, seed):
    """Held-out per-sample probability of gold via GroupKFold on item.

    Returns array proba aligned to X rows (each row scored by a fold that did NOT
    see its item). C chosen per outer-fold by an inner GroupKFold maximising AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    proba = np.full(len(y), np.nan)
    uniq_groups = np.unique(groups)
    n_splits = min(5, len(uniq_groups))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        # inner C-selection (maximise pooled option-level AUC, held-out within train)
        best_c, best_auc = Cs[0], -1.0
        gtr = groups[tr]
        inner_splits = min(3, len(np.unique(gtr)))
        if inner_splits >= 2 and len(Cs) > 1:
            for C in Cs:
                aucs = []
                for itr, ite in GroupKFold(n_splits=inner_splits).split(X[tr], y[tr], gtr):
                    sc = StandardScaler().fit(X[tr][itr])
                    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
                    clf.fit(sc.transform(X[tr][itr]), y[tr][itr])
                    p = clf.predict_proba(sc.transform(X[tr][ite]))[:, 1]
                    if len(np.unique(y[tr][ite])) > 1:
                        aucs.append(roc_auc_score(y[tr][ite], p))
                m = float(np.mean(aucs)) if aucs else -1.0
                if m > best_auc:
                    best_auc, best_c = m, C
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        proba[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return proba


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True,
                    help=".pt from extract_coldmc_perhead_pod.py")
    ap.add_argument("--anchor", type=Path, default=None,
                    help="offaxis_vs_stack_<task>.json for adapter/surface AUC anchors")
    ap.add_argument("--use-all-heads", action="store_true",
                    help="use acts_all (looser upper bound) instead of the adapter subset")
    ap.add_argument("--Cs", default="0.003,0.01,0.03,0.1,0.3",
                    help="L2 inverse-reg grid for the AUC-tuned probe")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if torch is None:
        raise SystemExit("torch required to load the .pt")

    st = torch.load(args.acts, weights_only=False, map_location="cpu")
    task = st["task"]
    print("=" * 80, flush=True)
    print(f"P1 ACTIVATION-PROBE IN-SPACE | task={task} | schema={st.get('schema_version')}",
          flush=True)
    print("=" * 80, flush=True)
    cells = st.get("adapter_cells_layer_head", [])
    print(f"  same-space cells (adapter layer,head): {len(cells)} | "
          f"items={st['n_items']} base-wrong={st['n_base_wrong']}", flush=True)

    if args.use_all_heads and "acts_all" in st:
        A = st["acts_all"]                       # (n_prompts, n_layers, n_heads, head_dim)
        acts = A.reshape(A.shape[0], -1).numpy().astype(np.float64)
        print(f"  [--use-all-heads] feature dim = {acts.shape[1]} (every captured head)", flush=True)
    else:
        A = st["acts_adapter"]                   # (n_prompts, n_cells, head_dim)
        acts = A.reshape(A.shape[0], -1).numpy().astype(np.float64)
        print(f"  feature dim = {acts.shape[1]} (adapter same-space cells x head_dim)", flush=True)

    meta = st["prompt_meta"]
    items = st["items"]
    bw_items = {it["item_idx"] for it in items if it["base_correct"] == 0}

    # rows for the probe: every (item, option), label=is_gold, group=item
    X = acts
    y = np.array([m["is_gold"] for m in meta], dtype=np.int64)
    groups = np.array([m["item_idx"] for m in meta], dtype=np.int64)
    print(f"  probe rows: {len(y)} ({int(y.sum())} gold / {len(y)-int(y.sum())} distractor)",
          flush=True)

    Cs = [float(x) for x in args.Cs.split(",") if x.strip()]
    try:
        proba = fit_heldout_scores(X, y, groups, Cs, args.seed)
        backend = "sklearn LogisticRegression (held-out GroupKFold by item, AUC-tuned C)"
    except Exception as e:  # pragma: no cover - fallback
        print(f"  [WARN] sklearn path failed ({type(e).__name__}: {e}); using numpy ridge-logit",
              flush=True)
        proba = _numpy_fallback(X, y, groups, args.seed)
        backend = "numpy ridge-logistic fallback (held-out by item)"
    print(f"  probe: {backend}", flush=True)

    # assemble per-item held-out score vectors, restricted to BASE-WRONG
    by_item_all, by_item_bw = [], []
    for it in items:
        idxs = it["prompt_idxs"]
        s = np.array([proba[i] for i in idxs], dtype=np.float64)
        gold = it["target"]
        if gold >= len(s):
            continue
        by_item_all.append((s, gold))
        if it["item_idx"] in bw_items:
            by_item_bw.append((s, gold))

    auc_all = pair_auc(by_item_all)
    auc_bw = pair_auc(by_item_bw)
    lo_bw, hi_bw = boot_pair_auc_ci(by_item_bw, args.n_boot, args.seed)

    # anchors
    anchor = args.anchor or (args.acts.parent / f"offaxis_vs_stack_{task}.json")
    adapter_auc = surface_auc = None
    if anchor.exists():
        aj = json.load(anchor.open(encoding="utf-8")).get("auc_base_wrong", {})
        adapter_auc = aj.get("adapter")
        surface_auc = aj.get("oracle_stack")

    print(f"\n  === pair_auc(gold>distractor) ===", flush=True)
    print(f"      all items        AUC={auc_all:.3f}  (n={len(by_item_all)})", flush=True)
    print(f"      BASE-WRONG       AUC={auc_bw:.3f}  CI[{lo_bw:.3f},{hi_bw:.3f}]  "
          f"(n={len(by_item_bw)})  <- compare here", flush=True)
    if surface_auc is not None:
        print(f"      [anchor] surface stack cap   {surface_auc:.3f}", flush=True)
    if adapter_auc is not None:
        print(f"      [anchor] adapter (off-axis)  {adapter_auc:.3f}", flush=True)
    print(f"      [anchor] W_know on-axis recovery 6.4% (LDA lower bound)", flush=True)

    # verdict
    verdict = "INDETERMINATE"
    if adapter_auc:
        if auc_bw >= adapter_auc - 0.05:
            verdict = (f"PURE MISCALIBRATION: same-space linear probe recovers {auc_bw:.3f} ~ "
                       f"adapter {adapter_auc:.3f} -> the correctness signal is present in the "
                       f"adapter's own subspace and ANY same-space reader extracts it. §5 leads "
                       f"with 'lossy readout, signal intact upstream' (strongest form).")
        elif auc_bw <= (surface_auc or 0.574) + 0.05:
            verdict = (f"REPRESENTATIONAL EDIT: probe {auc_bw:.3f} ~ surface cap "
                       f"{surface_auc or 0.574:.3f} << adapter {adapter_auc:.3f} -> a linear "
                       f"same-space reader CANNOT read what the offset writes; the adapter adds/"
                       f"rotates signal (stronger-and-different claim, not mere re-calibration).")
        else:
            verdict = (f"PARTIAL: probe {auc_bw:.3f} sits between surface cap "
                       f"{surface_auc or 0.574:.3f} and adapter {adapter_auc:.3f} -> the signal "
                       f"is partly linearly recoverable same-space; report the fraction, hedge "
                       f"the 'pure miscalibration' wording.")
    print(f"\n  === VERDICT (prose-freeze gate) ===\n      {verdict}", flush=True)

    payload = dict(task=task, feature_dim=int(X.shape[1]),
                   n_items=len(by_item_all), n_base_wrong=len(by_item_bw),
                   auc_all=auc_all, auc_base_wrong=auc_bw,
                   auc_base_wrong_ci=[lo_bw, hi_bw],
                   anchor_adapter=adapter_auc, anchor_surface=surface_auc,
                   probe=backend, cells=len(cells), verdict=verdict)
    outp = args.out or (args.acts.parent / f"activation_probe_inspace_{task}.json")
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {outp}", flush=True)


def _numpy_fallback(X, y, groups, seed):
    """Minimal held-out logistic via GD with L2, GroupKFold-by-item by hand."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, min(5, len(uniq)))
    proba = np.full(len(y), 0.5)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        w = np.zeros(Xs.shape[1]); b = 0.0
        lr, lam = 0.1, 1.0
        Xtr, ytr = Xs[tr], y[tr]
        for _ in range(300):
            z = Xtr @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - ytr
            w -= lr * (Xtr.T @ g / len(ytr) + lam * w / len(ytr))
            b -= lr * g.mean()
        proba[te] = 1.0 / (1.0 + np.exp(-(Xs[te] @ w + b)))
    return proba


if __name__ == "__main__":
    main()
