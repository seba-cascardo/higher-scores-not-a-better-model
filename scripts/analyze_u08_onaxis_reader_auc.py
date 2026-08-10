"""U-08(a) — are the on-axis directions bad READERS, or do they read fine and fail causally?

The paper's on-axis null (W_know recovers 6.4% of the ARC lift) is a CAUSAL result:
inject the direction, measure the lift. The adversarial review (U-08) points out that
this cannot distinguish

  (H1) the on-axis direction is a fine correctness discriminant and injecting it
       simply does not buy the lift  -> the null is about causality, and stands;
  (H2) the ESTIMATED on-axis direction is a noisy discriminant (N=2400 vs d=256,
       median cond ~894, whitening amplifies the worst-estimated eigendirections)
       -> the null may be about estimation error, and is confounded.

H1 and H2 differ in an observable the paper never reports: how well those exact
directions READ correctness in the same population where the fitted probe gets
pair-AUC 0.9545 (runs/vinf_causal/activation_probe_inspace_arc_challenge.json).

This script measures it, on CPU, from artefacts that already exist. The reader for an
offset blob is the linear functional whose 12288-dim weight vector IS the blob:

    score(option) = sum_h  alpha_h * <a_h(option), theta_h>

i.e. exactly the concatenation of the per-head injected vectors. Metric, population and
bootstrap are IMPORTED from scripts/analyze_activation_probe_inspace.py, so the numbers
are directly comparable to 0.9545.

NULL: random unit direction per head with the SAME per-head norms (alpha_h), redrawn
n_null times. This is the reader-side twin of the paper's random norm-matched injection
control. It preserves the blob's per-head energy budget and the activation distribution;
it destroys only the choice of direction. (It is a re-randomisation of the WEIGHTS, not
a permutation of the data, so it says nothing about the labels' marginal law.)

  python scripts/analyze_u08_onaxis_reader_auc.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.analyze_activation_probe_inspace import pair_auc, boot_pair_auc_ci  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load_blob(path, expect_key=None):
    d = torch.load(path, weights_only=False, map_location="cpu")
    cfg = d.get("config", {})
    if expect_key is not None and expect_key not in cfg:
        raise SystemExit(f"FATAL: {path} config lacks {expect_key!r} — wrong artefact "
                         f"(identify by CONTENT). config keys: {sorted(cfg)}")
    return d, cfg


def _weight_vector(blob, pairs_ref):
    """Concatenate the APPLIED per-head offset alpha_h * theta_h, verbatim.

    Not unit-renormalised: make_wknow_offset.py / make_vinf_offset.py match their
    blob to ``||alpha_mc * theta_mc||`` per head, so the applied vectors already
    carry identical per-head norms. Renormalising theta would silently re-weight
    the heads of the mc blob only (its theta is not unit) and destroy that match.
    """
    pairs = [tuple(int(x) for x in p) for p in blob["layer_head_pairs"]]
    if pairs != pairs_ref:
        raise SystemExit("FATAL: blob cell order differs from the activations' "
                         "adapter_cells_layer_head — refusing to score misaligned heads.")
    a = blob["alpha"].to(torch.float64)
    t = blob["theta"].to(torch.float64)
    return (a[:, None] * t).reshape(-1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", default="runs/vinf_causal/coldmc_perhead_arc_challenge.pt")
    ap.add_argument("--mc", default="runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt")
    ap.add_argument("--vinf", default="runs/vinf_causal/mc_vinf_offset.pt")
    ap.add_argument("--wknow", default="runs/vinf_causal/mc_wknow_offset.pt")
    ap.add_argument("--out", default="runs/vinf_causal/u08_onaxis_reader_auc.json")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 74)
    print("U-08(a) — on-axis directions as READERS on the probe's own population")
    print("=" * 74, flush=True)

    print(f"[1/4] loading {args.acts} (~2.7 GB) ...", flush=True)
    st = torch.load(args.acts, weights_only=False, map_location="cpu")
    print(f"      task={st['task']!r} schema={st.get('schema_version')} "
          f"n_items={st['n_items']} n_base_wrong={st['n_base_wrong']}")
    if st["task"] != "arc_challenge":
        raise SystemExit(f"FATAL: task is {st['task']!r}, not arc_challenge.")
    cells = [tuple(int(x) for x in c) for c in st["adapter_cells_layer_head"]]
    A = st["acts_adapter"]                                  # [n_rows, 48, 256]
    X = A.reshape(A.shape[0], -1).numpy().astype(np.float64)
    items = st["items"]
    bw = {it["item_idx"] for it in items if it["base_correct"] == 0}
    print(f"      cells={len(cells)}  rows={X.shape[0]}  dim={X.shape[1]}  "
          f"base-wrong items={len(bw)}", flush=True)

    print("[2/4] building reader weight vectors ...", flush=True)
    mcb, _ = _load_blob(args.mc)
    vb, vcfg = _load_blob(args.vinf, "vinf_variant")
    wb, wcfg = _load_blob(args.wknow, "wknow_variant")
    print(f"      v_inf  variant={vcfg['vinf_variant']!r}")
    print(f"      W_know variant={wcfg['wknow_variant']!r} shrinkage={wcfg.get('shrinkage_frac')}")
    readers = {
        "theta_mc (the trained offset — off-axis)": _weight_vector(mcb, cells),
        "v_inf (mass-mean on-axis)": _weight_vector(vb, cells),
        "W_know (Fisher-LDA on-axis)": _weight_vector(wb, cells),
    }
    # Per-head applied norms of the mc blob — the norm budget every arm is matched to.
    perhead_norm = (mcb["alpha"].to(torch.float64)[:, None]
                    * mcb["theta"].to(torch.float64)).norm(dim=1).numpy()
    for k, w in readers.items():
        ph = np.linalg.norm(w.reshape(len(cells), -1), axis=1)
        print(f"      {k:42s} ||w|| = {np.linalg.norm(w):8.4f}   "
              f"max per-head norm mismatch vs mc = {np.abs(ph - perhead_norm).max():.2e}")

    def auc_of(w, boot=False):
        s_all = X @ w
        by_all, by_bw = [], []
        for it in items:
            idxs = it["prompt_idxs"]
            gold = it["target"]
            s = s_all[idxs]
            if gold >= len(s):
                continue
            by_all.append((s, gold))
            if it["item_idx"] in bw:
                by_bw.append((s, gold))
        a_all, a_bw = pair_auc(by_all), pair_auc(by_bw)
        ci = boot_pair_auc_ci(by_bw, args.n_boot, args.seed) if boot else (None, None)
        return a_all, a_bw, ci, len(by_bw)

    print("[3/4] scoring ...", flush=True)
    res = {}
    for name, w in readers.items():
        a_all, a_bw, ci, n_bw = auc_of(w, boot=True)
        res[name] = {"auc_all": a_all, "auc_base_wrong": a_bw,
                     "auc_base_wrong_ci95": list(ci), "n_base_wrong": n_bw}
        print(f"  {name:42s} pair-AUC all {a_all:.4f}   base-wrong {a_bw:.4f}  "
              f"CI[{ci[0]:.4f},{ci[1]:.4f}]", flush=True)

    print(f"[4/4] random norm-matched direction null ({args.n_null} draws) ...", flush=True)
    rng = np.random.default_rng(args.seed)
    n_cells, hd = len(cells), A.shape[2]
    null_bw, null_all = [], []
    for i in range(args.n_null):
        R = rng.standard_normal((n_cells, hd))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        w = (perhead_norm[:, None] * R).reshape(-1)
        a_all, a_bw, _, _ = auc_of(w)
        null_all.append(a_all)
        null_bw.append(a_bw)
        if (i + 1) % 50 == 0:
            print(f"      [{i+1}/{args.n_null}] null draws", flush=True)
    null_bw = np.array(null_bw)
    # Two-sided reference band on the AUC itself (the null is symmetric about 0.5
    # only up to sign, so report both the raw band and the |AUC-0.5| band).
    band = [float(np.percentile(null_bw, 2.5)), float(np.percentile(null_bw, 97.5))]
    band_abs97 = float(np.percentile(np.abs(null_bw - 0.5), 97.5)) + 0.5
    print(f"      null base-wrong AUC: mean {null_bw.mean():.4f} sd {null_bw.std(ddof=1):.4f} "
          f"band95 [{band[0]:.4f},{band[1]:.4f}]  |AUC-.5| p97.5 -> {band_abs97:.4f}")
    # Empirical one-sided exceedance of each reader against the random-direction null.
    exceed = {}
    for name, r in res.items():
        a = r["auc_base_wrong"]
        exceed[name] = float((null_bw >= a).mean())
        print(f"      P(null >= {a:.4f}) for {name:42s} = {exceed[name]:.4f}")

    # --- paired bootstrap: is W_know a WORSE reader than v_inf? (the review's test
    #     for whether the word "strongest" survives). Same items resampled for both.
    def _by_bw(w):
        s_all = X @ w
        out_ = []
        for it in items:
            if it["item_idx"] not in bw:
                continue
            s = s_all[it["prompt_idxs"]]
            if it["target"] >= len(s):
                continue
            out_.append((s, it["target"]))
        return out_

    bw_w, bw_v = _by_bw(readers["W_know (Fisher-LDA on-axis)"]), \
        _by_bw(readers["v_inf (mass-mean on-axis)"])
    rng2 = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.n_boot):
        idx = rng2.integers(0, len(bw_w), len(bw_w))
        diffs.append(pair_auc([bw_w[i] for i in idx]) - pair_auc([bw_v[i] for i in idx]))
    diffs = np.array(diffs)
    d_pt = pair_auc(bw_w) - pair_auc(bw_v)
    d_ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    print(f"\n  paired bootstrap W_know - v_inf (base-wrong AUC): {d_pt:+.4f}  "
          f"CI95 [{d_ci[0]:+.4f},{d_ci[1]:+.4f}]  "
          f"P(W_know < v_inf) = {float((diffs < 0).mean()):.3f}")

    probe_ref = None
    p = Path("runs/vinf_causal/activation_probe_inspace_arc_challenge.json")
    if p.exists():
        probe_ref = json.loads(p.read_text(encoding="utf-8"))
        print(f"\n  [anchor] fitted linear probe, SAME space/population: "
              f"base-wrong AUC {probe_ref['auc_base_wrong']:.4f} "
              f"CI{[round(x,4) for x in probe_ref['auc_base_wrong_ci']]}")

    out = {
        "acts": args.acts, "task": st["task"], "n_items": st["n_items"],
        "n_base_wrong": st["n_base_wrong"], "cells": len(cells), "dim": int(X.shape[1]),
        "metric": "pair_auc(gold>distractor), imported from analyze_activation_probe_inspace",
        "reader_definition": "score = sum_h alpha_h * <a_h, unit(theta_h)>; the weight "
                             "vector IS the injected offset blob",
        "readers": res,
        "paired_wknow_minus_vinf": {"point": d_pt, "boot_ci95": d_ci,
                                    "p_wknow_worse": float((diffs < 0).mean()),
                                    "n_boot": args.n_boot},
        "null": {"kind": "random unit direction per head, same per-head norms alpha_h; "
                         "re-randomises the WEIGHTS, not the labels",
                 "n_draws": args.n_null, "mean": float(null_bw.mean()),
                 "sd": float(null_bw.std(ddof=1)), "band95_base_wrong": band,
                 "abs_dev_p975_as_auc": band_abs97,
                 "all_items_mean": float(np.mean(null_all)),
                 "p_null_ge_reader": exceed,
                 "draws_base_wrong": null_bw.tolist()},
        "anchor_fitted_probe": probe_ref,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
