"""U-08(b) — build the PROBE-DIRECTION on-axis offset, norm-matched to the mc offset.

The paper's on-axis arms use directions estimated OFF the evaluation task (v_inf,
W_know come from the functional-axis corpus). The adversarial review (U-08b) asks for
the arm that closes the estimation loophole outright: inject the *empirically optimal*
linear correctness reader in the adapter's own space -- the fitted probe direction,
pair-AUC ~0.955 -- through the identical norm-matched harness.

This script builds that blob (CPU, EUR 0). It does NOT evaluate it: measuring what an
injected offset DOES requires forward passes through the 31B, i.e. a GPU run.

CIRCULARITY GUARD -- load-bearing, and the reason for --train-until:
  v_inf/W_know are fit on a different corpus, so evaluating them on ARC is clean.
  The probe is fit ON ARC. A probe fit on the same items it is later evaluated on
  would inject label information. So the blob is fit on the 400-item development
  slice ONLY (item_idx < 400) and the pod evaluation must score the COMPLEMENT
  (ARC-Challenge test items 400..1171). The out-of-sample reader AUC printed here is
  the pre-check that the direction is still a strong reader where it will be judged.

  python scripts/make_probe_offset.py
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

from scripts.analyze_probe_direction_geometry import fit_probe_dir, whiten_M, cos_raw, cos_whit  # noqa: E402
from scripts.analyze_activation_probe_inspace import pair_auc, boot_pair_auc_ci  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", default="runs/vinf_causal/coldmc_perhead_arc_challenge_full.pt")
    ap.add_argument("--mc", default="runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt")
    ap.add_argument("--func", default="runs/functional_directions.pt")
    ap.add_argument("--wknow", default="runs/vinf_causal/mc_wknow_offset.pt")
    ap.add_argument("--train-until", type=int, default=400,
                    help="fit the probe on item_idx < N; score the complement")
    ap.add_argument("--C", type=float, default=0.03)
    ap.add_argument("--pca", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/vinf_causal/mc_probe_offset.pt")
    ap.add_argument("--report", default="runs/vinf_causal/mc_probe_offset_report.json")
    args = ap.parse_args()

    print("=" * 74)
    print("U-08(b) — probe-direction on-axis offset (build only; eval needs a GPU)")
    print("=" * 74, flush=True)

    st = torch.load(args.acts, weights_only=False, map_location="cpu")
    if st["task"] != "arc_challenge":
        raise SystemExit(f"FATAL: task={st['task']!r}")
    A = st["acts_adapter"]
    nP, nC, hd = A.shape
    X = A.reshape(nP, nC * hd).numpy().astype(np.float64)
    y = np.array([m["is_gold"] for m in st["prompt_meta"]], dtype=np.int64)
    groups = np.array([m["item_idx"] for m in st["prompt_meta"]], dtype=np.int64)
    cells = [tuple(int(z) for z in c) for c in st["adapter_cells_layer_head"]]
    items = st["items"]
    print(f"  acts: {args.acts}  items={st['n_items']} rows={nP} cells={nC} hd={hd}")

    tr = groups < args.train_until
    te_items = [it for it in items if it["item_idx"] >= args.train_until]
    print(f"  fit on {int(tr.sum())} rows / {len(np.unique(groups[tr]))} items "
          f"(idx < {args.train_until}); held out {len(te_items)} items", flush=True)

    w_raw, _, _, _ = fit_probe_dir(X[tr], y[tr], args.C, args.pca, args.seed)
    P = w_raw.reshape(nC, hd)
    P_unit = P / np.linalg.norm(P, axis=1, keepdims=True)

    mc = torch.load(args.mc, weights_only=False, map_location="cpu")
    mc_pairs = [tuple(int(z) for z in p) for p in mc["layer_head_pairs"]]
    if mc_pairs != cells:
        raise SystemExit("FATAL: cell order mismatch acts vs offsets_mc")
    alpha_mc = mc["alpha"].to(torch.float64).numpy()
    theta_mc = mc["theta"].to(torch.float64).numpy()
    applied_mc = alpha_mc[:, None] * theta_mc
    scale = np.linalg.norm(applied_mc, axis=1)                 # norm budget per head

    new_alpha = torch.tensor(scale, dtype=torch.float32)
    new_theta = torch.tensor(P_unit, dtype=torch.float32)
    blob = {**mc, "alpha": new_alpha, "theta": new_theta,
            "config": {**mc.get("config", {}),
                       "probe_variant": "mc_probe_direction_onaxis",
                       "built_from": {"mc": args.mc, "acts": args.acts},
                       "probe": {"C": args.C, "pca": args.pca, "seed": args.seed,
                                 "train_until": args.train_until,
                                 "fit_rows": int(tr.sum())},
                       "norm_matched_to": "applied_mc_offset",
                       "eval_constraint": f"MUST be scored on ARC items >= "
                                          f"{args.train_until} (probe fit on the rest)"}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    chk = torch.load(args.out, weights_only=False, map_location="cpu")
    applied_new = (chk["alpha"].to(torch.float64)[:, None] * chk["theta"].to(torch.float64)).numpy()
    print(f"  norm-match max abs err: "
          f"{np.abs(np.linalg.norm(applied_new, axis=1) - scale).max():.2e}", flush=True)

    # --- geometry vs the arms already run -----------------------------------------
    fd = torch.load(args.func, weights_only=False, map_location="cpu")
    v_inf = fd["v_inf"].to(torch.float64).numpy()
    cap = {L: ai for ai, L in enumerate(int(z) for z in fd["captured_indices"])}
    cov_inf = fd["cov_inf"]
    wk = torch.load(args.wknow, weights_only=False, map_location="cpu")
    W = (wk["alpha"].to(torch.float64)[:, None] * wk["theta"].to(torch.float64)).numpy()

    g = {"raw_cos_probe_theta": [], "raw_cos_probe_vinf": [], "raw_cos_probe_wknow": [],
         "whit_cos_probe_theta": [], "whit_cos_probe_vinf": [], "whit_cos_probe_wknow": []}
    for i, (L, h) in enumerate(cells):
        p_ = P_unit[i]
        g["raw_cos_probe_theta"].append(cos_raw(p_, theta_mc[i]))
        g["raw_cos_probe_wknow"].append(cos_raw(p_, W[i]))
        if L in cap:
            g["raw_cos_probe_vinf"].append(cos_raw(p_, v_inf[cap[L], h]))
        cov = cov_inf.get((L, h))
        if cov is None or L not in cap:
            continue
        M = whiten_M(cov.to(torch.float64).numpy(), hd)
        g["whit_cos_probe_theta"].append(cos_whit(p_, theta_mc[i], M))
        g["whit_cos_probe_vinf"].append(cos_whit(p_, v_inf[cap[L], h], M))
        g["whit_cos_probe_wknow"].append(cos_whit(p_, W[i], M))
    geo = {k: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
               "absmean": float(np.mean(np.abs(v)))} for k, v in g.items() if v}
    print("\n  geometry of the probe direction (per head):")
    for k, s in geo.items():
        print(f"    {k:24s} mean {s['mean']:+.4f}  |mean| {s['absmean']:.4f}  (n={s['n']})")

    # --- out-of-sample reader AUC of the blob as built ------------------------------
    w_applied = applied_new.reshape(-1)
    s_all = X @ w_applied
    by_bw = []
    for it in te_items:
        if it["base_correct"] != 0:
            continue
        s = s_all[it["prompt_idxs"]]
        if it["target"] >= len(s):
            continue
        by_bw.append((s, it["target"]))
    auc = pair_auc(by_bw)
    lo, hi = boot_pair_auc_ci(by_bw, args.n_boot, args.seed)
    print(f"\n  OUT-OF-SAMPLE reader check (items >= {args.train_until}, base-wrong "
          f"n={len(by_bw)}): pair-AUC {auc:.4f}  CI[{lo:.4f},{hi:.4f}]")

    rep = {"out": args.out, "acts": args.acts, "train_until": args.train_until,
           "C": args.C, "pca": args.pca, "seed": args.seed,
           "n_cells": nC, "geometry_per_head": geo,
           "oos_reader": {"n_base_wrong": len(by_bw), "pair_auc": auc, "ci95": [lo, hi]},
           "eval_constraint": blob["config"]["eval_constraint"]}
    Path(args.report).write_text(json.dumps(rep, indent=2))
    print(f"\n  wrote {args.out}\n  wrote {args.report}")
    print("\n  NEXT (GPU only): score this blob through the same harness as the other "
          "injection arms, restricted to ARC items >= "
          f"{args.train_until}.")


if __name__ == "__main__":
    main()
