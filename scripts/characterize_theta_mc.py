"""Characterize what direction the trained mc offset (theta_mc) actually pushes.

E2 (probe_oq1_functional_angle) found cos_whitened(v_inf, theta_mc) ≈ 0: the trained
offset is ORTHOGONAL to the functional inference-correctness direction. Since v_inf is
itself aligned with v_ret (E2, cos_w +0.283), theta_mc is then likely orthogonal to
BOTH correctness directions. So WHAT does the adapter push along? This decides the
mechanism behind the scoring lift — and whether process-LoFiT-PRM (push along v_inf
instead) is the right move.

LOCAL, CPU, €0. Uses ONLY what extract_functional_axis_perhead.py already saved in
functional_directions.pt — no model, no pod. Per adapter head (theta_mc != 0), projects
theta_mc onto a battery of candidate directions:
  - v_inf, v_ret                : functional correctness directions (E2). cos≈0 ⇒ not correctness.
  - mu_correct, mu_wrong        : per-class activation centroids (both families).
  - centroid = (mu_c + mu_w)/2  : the NON-discriminative mean shift (does the offset
                                   just push toward the average activation?).
  - top-k PCs of base cov       : high-variance subspace (format/style energy).
All raw AND whitened (Mahalanobis under the per-head base covariance, same convention as
park_whiten / probe_oq1_functional_angle). Reports per-family mean cosine ± BCa over the
adapter heads, theta_mc's norm relative to the class gap ||mu_c - mu_w||, and the
fraction of theta_mc's energy living in the top-k PC subspace.

VERDICT:
  - |cos(theta_mc, v_inf)| and |cos(theta_mc, v_ret)| within null  → NOT correctness-aligned
    (confirms + extends E2: the offset is orthogonal to both correctness axes).
  - high cos(theta_mc, centroid)                                    → MEAN SHIFT (pushes toward
    the average correct/overall activation, not a discriminative direction).
  - high top-k-PC energy fraction                                   → HIGH-VARIANCE push
    (format/style/confidence subspace, not correctness).
  - none of the above                                              → UNEXPLAINED: theta_mc lies
    off every available axis → needs NEW candidate directions via fresh extraction
    (confidence probe, option-position bias, length) — flagged as next experiment.

Usage (CPU, after E2's .pt is local):
  python -m scripts.characterize_theta_mc \\
      --acts runs/functional_directions.pt \\
      --top-pc 10 \\
      --out runs/oq1_functional_axis/theta_mc_characterization.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# repo root on path so `from scripts.X` resolves under both invocation styles.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the EXACT whitening + cos used by the E2 probe (Mahalanobis via eigh of the
# per-head pooled covariance, eigvals clamped at 1e-6). Importing keeps the geometry
# byte-identical to the angle that produced the ALIGNED verdict.
from scripts.probe_oq1_functional_angle import _whitening_from_cov, _whiten_vec, _cos


def _bca_lite(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean (lite; the E2 probe carries the full BCa,
    here we only need a CI band over the adapter heads to size the effect)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n < 3:
        m = float(v.mean())
        return (m, m, m)
    rng = np.random.default_rng(seed)
    boot = np.array([v[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return (float(v.mean()),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def _summ(vals):
    a = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def _grid(g, ai, hi):
    return g[ai, hi, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", default="runs/functional_directions.pt",
                    help="The .pt from extract_functional_axis_perhead.py (schema oq1-func-1.0).")
    ap.add_argument("--out", default="runs/oq1_functional_axis/theta_mc_characterization.json")
    ap.add_argument("--top-pc", type=int, default=10,
                    help="Top-k PCs of the per-head base covariance for the energy fraction.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("theta_mc characterization — what does the trained offset push along?")
    print("=" * 64)
    print(f"  acts: {args.acts}", flush=True)

    payload = torch.load(args.acts, weights_only=False, map_location="cpu")
    for k in ("theta_mc", "v_inf", "v_ret"):
        if k not in payload:
            print(f"FATAL: .pt lacks '{k}' — not an oq1-func-1.0 extractor output.",
                  file=sys.stderr)
            sys.exit(2)

    theta = payload["theta_mc"]                 # [n_lay, nH, D] (alpha*theta, applied)
    v_inf, v_ret = payload["v_inf"], payload["v_ret"]
    mu_ic = payload.get("mu_inf_correct"); mu_iw = payload.get("mu_inf_wrong")
    mu_rc = payload.get("mu_ret_correct"); mu_rw = payload.get("mu_ret_wrong")
    cov_inf = payload.get("cov_inf", {}) or {}
    cov_ret = payload.get("cov_ret", {}) or {}
    captured = [int(x) for x in payload["captured_indices"]]
    n_lay, n_heads = theta.shape[0], theta.shape[1]

    def _cov_lookup(d, li, hi):
        return d.get((li, hi), d.get((int(li), int(hi))))

    recs = []
    n_adapter = 0
    for ai in range(n_lay):
        li = captured[ai]
        for hi in range(n_heads):
            t = _grid(theta, ai, hi)
            if float(torch.linalg.norm(t)) < 1e-12:
                continue                         # not an adapter head
            n_adapter += 1
            vi = _grid(v_inf, ai, hi)
            vr = _grid(v_ret, ai, hi)
            rec = {"layer": li, "head": hi,
                   "cos_t_vinf_raw": _cos(t, vi),
                   "cos_t_vret_raw": _cos(t, vr)}

            # class centroids + non-discriminative mean shift (inference family)
            if mu_ic is not None and mu_iw is not None:
                c, w = _grid(mu_ic, ai, hi), _grid(mu_iw, ai, hi)
                centroid = 0.5 * (c + w)
                gap = c - w
                rec["cos_t_mu_inf_correct_raw"] = _cos(t, c)
                rec["cos_t_centroid_inf_raw"] = _cos(t, centroid)
                gn = float(torch.linalg.norm(gap))
                rec["norm_ratio_inf"] = (float(torch.linalg.norm(t)) / gn) if gn > 1e-12 else float("nan")
            if mu_rc is not None and mu_rw is not None:
                c, w = _grid(mu_rc, ai, hi), _grid(mu_rw, ai, hi)
                rec["cos_t_centroid_ret_raw"] = _cos(t, 0.5 * (c + w))

            # whitened cosines + top-PC energy, only where covariance is stored.
            ci, cr = _cov_lookup(cov_inf, li, hi), _cov_lookup(cov_ret, li, hi)
            if ci is not None and cr is not None:
                cov_common = 0.5 * (ci.to(torch.float64) + cr.to(torch.float64))
                W, eig = _whitening_from_cov(cov_common)
                tw = _whiten_vec(W, t)
                rec["cos_t_vinf_whitened"] = _cos(tw, _whiten_vec(W, vi))
                rec["cos_t_vret_whitened"] = _cos(tw, _whiten_vec(W, vr))
                if mu_ic is not None:
                    cen = 0.5 * (_grid(mu_ic, ai, hi) + _grid(mu_iw, ai, hi))
                    rec["cos_t_centroid_inf_whitened"] = _cos(tw, _whiten_vec(W, cen))
                # top-k PC energy fraction of theta_mc in the head's base cov eigenbasis
                eigvals, eigvecs = torch.linalg.eigh(ci.to(torch.float64))
                order = torch.argsort(eigvals, descending=True)
                top = eigvecs[:, order[:args.top_pc]]               # [D, k]
                t64 = t.to(torch.float64)
                proj = top.T @ t64                                  # [k]
                tot = float((t64 @ t64))
                rec["topk_pc_energy_frac"] = float((proj @ proj) / tot) if tot > 1e-18 else float("nan")
            recs.append(rec)

    if n_adapter == 0:
        print("FATAL: no adapter heads (theta_mc all zero). Wrong .pt or alignment failed.",
              file=sys.stderr)
        sys.exit(2)

    print(f"  adapter heads with theta_mc != 0: {n_adapter}", flush=True)

    # aggregate
    keys = ["cos_t_vinf_raw", "cos_t_vret_raw", "cos_t_vinf_whitened", "cos_t_vret_whitened",
            "cos_t_mu_inf_correct_raw", "cos_t_centroid_inf_raw", "cos_t_centroid_ret_raw",
            "cos_t_centroid_inf_whitened", "norm_ratio_inf", "topk_pc_energy_frac"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in recs if k in r and np.isfinite(r[k])]
        if vals:
            m, lo, hi = _bca_lite(np.array(vals), args.n_boot, args.seed)
            agg[k] = {**_summ(vals), "ci95": [lo, hi]}

    def g(k):
        return agg.get(k, {}).get("mean")

    print("\n  --- aggregate over adapter heads ---")
    for k in keys:
        if k in agg:
            a = agg[k]
            print(f"    {k:32s} mean {a['mean']:+.4f}  CI[{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}]  (n={a['n']})")

    # verdict
    cos_vinf = g("cos_t_vinf_whitened") if "cos_t_vinf_whitened" in agg else g("cos_t_vinf_raw")
    cos_vret = g("cos_t_vret_whitened") if "cos_t_vret_whitened" in agg else g("cos_t_vret_raw")
    cos_cen = g("cos_t_centroid_inf_whitened") if "cos_t_centroid_inf_whitened" in agg else g("cos_t_centroid_inf_raw")
    pc_frac = g("topk_pc_energy_frac")
    null_hi = 0.05  # rough |cos| null band for "≈ 0" at these dims; E2 mean-null was ±0.007

    reasons = []
    if cos_vinf is not None and abs(cos_vinf) < null_hi and (cos_vret is None or abs(cos_vret) < null_hi):
        reasons.append(f"theta_mc ⊥ BOTH correctness directions (cos_vinf {cos_vinf:+.3f}, "
                       f"cos_vret {cos_vret if cos_vret is None else round(cos_vret,3)}) → NOT a correctness sharpener.")
    if cos_cen is not None and abs(cos_cen) >= 0.20:
        reasons.append(f"theta_mc aligns with the class centroid (cos {cos_cen:+.3f}) → MEAN SHIFT "
                       f"toward the average activation, not a discriminative push.")
    if pc_frac is not None and pc_frac >= 0.5:
        reasons.append(f"{pc_frac:.0%} of theta_mc energy is in the top-{args.top_pc} PCs of base cov "
                       f"→ HIGH-VARIANCE (format/style/confidence) subspace.")
    if not reasons:
        reasons.append("theta_mc lies off every available axis (not v_inf/v_ret, not the centroid, "
                       "not the top PCs) → UNEXPLAINED; needs NEW candidate directions via extraction "
                       "(confidence probe / option-position bias / length).")

    verdict = " ".join(reasons)
    print("\n" + "=" * 64)
    print("VERDICT:")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 64, flush=True)

    out = {
        "acts": str(args.acts),
        "n_adapter_heads": n_adapter,
        "top_pc": args.top_pc,
        "aggregate": agg,
        "verdict": verdict,
        "records": recs,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {out_path}")
    print(json.dumps({
        "verdict": verdict,
        "cos_t_vinf": cos_vinf, "cos_t_vret": cos_vret,
        "cos_t_centroid": cos_cen, "topk_pc_energy_frac": pc_frac,
        "n_adapter_heads": n_adapter,
    }, indent=2))


if __name__ == "__main__":
    main()
