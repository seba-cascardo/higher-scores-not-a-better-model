"""U-01 — cos_whitened(theta_mc, v_ret) with the SAME estimator that produced -0.003.

The paper reports cos_w(theta_mc, v_inf) = -0.003 and cos_w(v_inf, v_ret) = +0.283
(runs/oq1_functional_axis/oq1_functional_angle.json, produced by
scripts/probe_oq1_functional_angle.py). It never reports cos(theta_mc, v_ret), yet
sections/discussion.tex asserts the offset is off-axis to BOTH faces. This script
measures the missing quantity, reusing the canonical estimator by IMPORT (not by
re-implementation) so the two numbers are comparable:

  - same eligibility rule  : acc_inf >= t AND acc_ret >= t, both families
  - same whitening metric  : W = (0.5*(cov_inf + cov_ret))^{-1/2}, park_whiten math
  - same aggregation       : mean over heads with a non-zero theta_mc
  - same null              : random unit vectors in head_dim, band on the MEAN over
                             the SAME n (concentration of measure; NOT a permutation
                             of the data, so it preserves nothing about the data's
                             conditional law -- it is a pure isotropic reference)
  - same CI                : BCa bootstrap over heads

Also reports, as the decisive discriminant, cos_w(theta_mc, v_ret_perp), where
v_ret_perp is the component of v_ret orthogonal to v_inf IN THE WHITENED METRIC:
that is the part of the retrieval face that the -0.003 measurement could not have
seen, so it is where a deflationary "the offset pushes on retrieval" account has to
live.

  python scripts/analyze_u01_thetamc_vret.py \
      --acts runs/functional_directions.pt \
      --mc runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --out runs/oq1_functional_axis/u01_thetamc_vret.json
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

# Canonical math, imported verbatim from the script that produced -0.003 / +0.283.
from scripts.probe_oq1_functional_angle import (  # noqa: E402
    _whitening_from_cov, _whiten_vec, _cos, _grid_get, _summarize,
    random_unit_null_cos, bca_ci,
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", default="runs/functional_directions.pt")
    ap.add_argument("--mc", default="runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt")
    ap.add_argument("--out", default="runs/oq1_functional_axis/u01_thetamc_vret.json")
    ap.add_argument("--acc-threshold", type=float, default=0.60)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-null", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 68)
    print("U-01 — cos_whitened(theta_mc, v_ret), canonical estimator")
    print("=" * 68, flush=True)

    print(f"[1/5] loading {args.acts} (~179 MB) ...", flush=True)
    p = torch.load(args.acts, weights_only=False, map_location="cpu")
    print(f"      schema={p.get('schema_version')}  base={p.get('base')}")
    print(f"      chat_template={p.get('chat_template')}  "
          f"retrieval={p.get('retrieval_datasets_used')} "
          f"fallback={p.get('retrieval_fallback_fired')}", flush=True)

    v_inf, v_ret = p["v_inf"], p["v_ret"]
    acc_inf, acc_ret = p["acc_inf"], p["acc_ret"]
    theta_mc = p.get("theta_mc")
    if theta_mc is None:
        raise SystemExit("FATAL: no theta_mc grid in the .pt — cannot reproduce the -0.003 join.")
    captured = [int(x) for x in p["captured_indices"]]
    cov_inf = p.get("cov_inf", {}) or {}
    cov_ret = p.get("cov_ret", {}) or {}
    head_dim = int(p.get("head_dim", v_inf.shape[-1]))

    # --- integrity: the theta_mc grid must BE the trained offset (alpha*theta) ----
    print(f"[2/5] cross-checking theta_mc grid against {args.mc} ...", flush=True)
    mc = torch.load(args.mc, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in q) for q in mc["layer_head_pairs"]]
    alpha = mc["alpha"].to(torch.float64)
    theta = mc["theta"].to(torch.float64)
    cap = {li: ai for ai, li in enumerate(captured)}
    max_err, n_joined = 0.0, 0
    for i, (L, h) in enumerate(pairs):
        if L not in cap:
            continue
        g = _grid_get(theta_mc, cap[L], h).to(torch.float64)
        max_err = max(max_err, float((g - alpha[i] * theta[i]).abs().max()))
        n_joined += 1
    print(f"      joined {n_joined}/{len(pairs)} adapter heads; "
          f"max|grid - alpha*theta| = {max_err:.3e}", flush=True)

    # --- per-head whitened cosines over the SAME eligible population --------------
    print("[3/5] whitening per eligible head ...", flush=True)
    n_lay, n_heads = v_inf.shape[0], v_inf.shape[1]

    def _cov_lookup(d, li, hi):
        return d.get((li, hi), d.get((int(li), int(hi))))

    recs = []
    n_elig = 0
    done = 0
    for ai in range(n_lay):
        li = captured[ai]
        for hi in range(n_heads):
            done += 1
            if done % 500 == 0:
                print(f"      [{done}/{n_lay*n_heads}] heads processed", flush=True)
            if not (float(acc_inf[ai, hi]) >= args.acc_threshold
                    and float(acc_ret[ai, hi]) >= args.acc_threshold):
                continue
            n_elig += 1
            ci = _cov_lookup(cov_inf, li, hi)
            cr = _cov_lookup(cov_ret, li, hi)
            if ci is None or cr is None:
                continue
            tmc = _grid_get(theta_mc, ai, hi)
            if float(torch.linalg.norm(tmc)) <= 1e-12:
                continue                       # head not touched by the adapter
            vi = _grid_get(v_inf, ai, hi)
            vr = _grid_get(v_ret, ai, hi)
            W, _ = _whitening_from_cov(0.5 * (ci.to(torch.float64) + cr.to(torch.float64)))
            wi, wr, wt = _whiten_vec(W, vi), _whiten_vec(W, vr), _whiten_vec(W, tmc)
            # v_ret component orthogonal to v_inf, in the whitened metric.
            wi_u = wi / torch.linalg.norm(wi)
            wr_perp = wr - (wr @ wi_u) * wi_u
            recs.append({
                "layer": li, "head": hi,
                "cos_w_theta_vret": _cos(wt, wr),
                "cos_w_theta_vinf": _cos(wt, wi),
                "cos_w_vinf_vret": _cos(wi, wr),
                "cos_w_theta_vret_perp": _cos(wt, wr_perp),
                "theta_norm": float(torch.linalg.norm(tmc)),
            })

    n = len(recs)
    print(f"      eligible heads (acc>={args.acc_threshold} both families): {n_elig}")
    print(f"      of those, heads with a non-zero theta_mc: {n}", flush=True)
    if n == 0:
        raise SystemExit("FATAL: no head carries both eligibility and a theta_mc.")

    # --- null band: mean over n iid random-unit cosines in R^head_dim -------------
    print("[4/5] null band + BCa ...", flush=True)
    null = random_unit_null_cos(head_dim, args.n_null, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    null_means = np.array([np.mean(rng.choice(null, size=n, replace=True))
                           for _ in range(20000)])
    band = [float(np.percentile(null_means, 2.5)), float(np.percentile(null_means, 97.5))]
    null_abs = np.array([np.mean(np.abs(rng.choice(null, size=n, replace=True)))
                         for _ in range(20000)])
    band_abs = [float(np.percentile(null_abs, 2.5)), float(np.percentile(null_abs, 97.5))]

    out = {
        "acts": args.acts, "mc": args.mc,
        "schema_version": p.get("schema_version"), "base": p.get("base"),
        "chat_template": p.get("chat_template"),
        "retrieval_datasets_used": p.get("retrieval_datasets_used"),
        "retrieval_fallback_fired": p.get("retrieval_fallback_fired"),
        "acc_threshold": args.acc_threshold,
        "head_dim": head_dim,
        "n_eligible": n_elig,
        "n_heads_measured": n,
        "theta_grid_vs_offsets_mc_maxabs_err": max_err,
        "theta_grid_heads_joined": n_joined,
        "estimator": "cos(W v, W u), W = (0.5*(cov_inf+cov_ret))^{-1/2}; "
                     "imported from scripts/probe_oq1_functional_angle.py",
        "null": {"kind": "iid random unit vectors in R^head_dim (isotropic reference; "
                         "NOT a permutation of the data)",
                 "single_draw_sd": float(null.std(ddof=1)),
                 "band_mean_over_n": band, "band_mean_abs_over_n": band_abs, "n": n},
        "metrics": {}, "records": recs,
    }
    for key in ["cos_w_theta_vret", "cos_w_theta_vinf", "cos_w_vinf_vret",
                "cos_w_theta_vret_perp"]:
        vals = np.array([r[key] for r in recs], dtype=np.float64)
        lo, mean, hi = bca_ci(vals, args.n_boot, args.seed)
        s = _summarize(vals)
        s["bca95"] = [lo, hi]
        s["mean_abs"] = float(np.abs(vals).mean())
        out["metrics"][key] = s
        print(f"  {key:24s} mean {s['mean']:+.4f}  BCa95 [{lo:+.4f},{hi:+.4f}]  "
              f"mean|.| {s['mean_abs']:.4f}  (n={s['n']})", flush=True)
    print(f"  null band for the mean over n={n}: [{band[0]:+.4f},{band[1]:+.4f}]; "
          f"for the mean |.|: [{band_abs[0]:.4f},{band_abs[1]:.4f}]")

    print("[5/5] writing ...", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
