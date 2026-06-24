"""Decompose the trained off-axis offset theta_mc in KAPPA's {W_know, W_pred} basis.

================================================================================
WHAT THIS IS  (the MECHANISM experiment; €0 CPU-local, pure geometry)
================================================================================
GATE-2 has two heads (battery spec, kill-rule BICEFALO). This is the artifact for
HEAD 2 (characterization / paper): *WHERE does the off-axis offset live relative to
KAPPA's geometry?* KAPPA (Park,Pyun,Jo, arXiv:2509.23782, ICML2026) posits two
per-head subspaces in the residual: a KNOWLEDGE direction W_know (what the model
knows) and a PREDICTION direction W_pred (what it predicts); KAPPA moves prediction
TOWARD knowledge. We already know cos(theta_mc, W_know) ~= 0.0014 (off-axis to
knowledge). This script measures where theta_mc sits w.r.t. the {W_know, W_pred}
plane, with the nulls a hostile reviewer would demand.

It reuses build_head_bases / _lda_direction / _top_pcs from score_mc_kappa_pod.py so
W_know / W_pred are BYTE-IDENTICAL to the GATE-2 KAPPA foil (no second estimator).
No model load, no forward passes: it operates on runs/functional_directions.pt
(theta_mc / mu_inf_* / cov_inf) in the per-head o_proj-input space (d=256) — the only
space where theta_mc, W_know, W_pred and the offset co-exist (forward-filter:
same-space). The adapter touches only ~48 heads (theta_mc nonzero); the decomposition
runs on exactly those.

================================================================================
DESIGN  (adversarially math-verified — fixes from the ClaraVT verify pass folded in)
================================================================================
Headline metric is R2_span (energy of unit theta_mc captured by span{W_know,W_pred}),
computed via a QR orthonormal basis of the plane — INVARIANT to the (oblique) choice
of basis, so it does NOT depend on the reconstructed W_pred the way a raw cosine does.

Per head (unit theta_hat, unit wk, unit wp):
  g           = wk . wp                       (signed obliquity)
  cond_G      = (1+|g|)/(1-|g|)               (2x2 Gram condition; checked BEFORE QR)
  R2_span     = ||Q^T theta_hat||^2,  Q = qr([wk|wp])      [INVARIANT; headline]
  R2_perp     = 1 - R2_span
  ASYMMETRIC partition (wk absorbs the wk.wp overlap; auditable, NOT the claim):
     R2_know_pure = (wk.theta_hat)^2
     wp_perp      = unit(wp - (wp.wk) wk);  R2_pred_pure = (wp_perp.theta_hat)^2
     assert R2_know_pure + R2_pred_pure == R2_span     (Pythagoras in orthonormal frame)
  SYMMETRIC complement (pred absorbs the overlap) — VERIFY-FIX [ALTA-1]:
     R2_pred_axis = (wp.theta_hat)^2
     wk_perp      = unit(wk - (wk.wp) wp);  R2_know_resid = (wk_perp.theta_hat)^2
     => Outcome-D (energy in knowledge) only fires if energy survives in BOTH partitions.
  OBLIQUE coords + Gram identity assert — VERIFY-FIX [MED]:
     c = G^-1 [wk.th, wp.th];  assert c^T G c == R2_span;  cross_term = 2 g c_know c_pred

Nulls (matched to THIS head's (wk,wp); fixed-seed random unit dirs in R^256):
  N1   E[(axis.rand)^2] ~ 1/d ; E[||Q^T rand||^2] ~ 2/d  (analytic Beta as 2nd check)
  C1 test (orthogonality to W_know) — VERIFY-FIX [ALTA-2]: LOW-tail p-value
     p_low_know = P(E_rand <= R2_know_pure). Small => theta has anomalously LITTLE
     knowledge energy (the actual orthogonality claim; the high-tail p cannot test it).
  N2a  theta FIXED vs RANDOM 2-plane: p_plane = P(R2_span(rand plane) >= R2_span)
  N2b  VERIFY-FIX [MED]: theta FIXED vs {PC1, RANDOM} plane — separates "the KAPPA
       plane is special" from "the plane merely contains the high-variance PC1".
  N3   VERIFY-FIX [MED]: R2 of theta on PC1 vs on PC2..PCk (rank-matched) — isolates
       "theta aligns with PC1 specifically" from "theta hits any high-variance axis".
       R2_pred_pure(pc1) is NEVER reported without this PC>=2 null beside it.

VERIFY-FIX [BAJA]: degenerate basis (cond_G > cond_max, e.g. wpred==wknow self-test)
is detected BEFORE the QR (a rank-deficient QR yields garbage) and the head is skipped.

CLAIM by outcome (battery spec head-2; wording is "where the DIRECTION lives", NOT
"what the offset does" — measuring W_o.theta would be the effect side, out of scope):
  A  R2_span ~= N2a/N2b floor & R2_perp high, stable across wpred (pc1 & vinf):
     HEADLINE (C1+C3 by exclusion) — theta_mc is orthogonal to W_know AND lives OUTSIDE
     the {W_know,W_pred} plane of KAPPA (the offset direction is not in either KAPPA axis).
  B  R2_span low but R2_pred_pure(pc1) survives N3 (>> PC>=2) & cos_know~0:
     EXPLORATORY (C4, "rewrites the prediction coord, the inverse of KAPPA") — caveated:
     W_pred is the reconstructed PC1, not the paper's greedy-option probe.
  C  R2_pred_pure diverges pc1 vs vinf (fails N3): C4 falls (proxy-dependent); C3 stands.
  D  R2_know energy >> floor in BOTH partitions: ALARM, contradicts off-axis. Audit
     cond_G / re-run with wk-orthogonalized / check cap_index before believing.

Usage (CPU-local, EUR0):
  python scripts/decompose_theta_mc_kappa_subspaces.py \
      --func runs/functional_directions.pt \
      --wpred-source pc1 --out runs/gate2/theta_decomp_pc1.json
  # then --wpred-source vinf  (M5 cross-proxy stability),
  # and  --wpred-source wknow (self-test: every head must be degenerate_basis).
Compile-check:  python -c "import ast;ast.parse(open('scripts/decompose_theta_mc_kappa_subspaces.py').read())"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Reuse the EXACT subspace builders + grid logic from the KAPPA foil so W_know/W_pred
# are byte-identical to GATE-2 (no second estimator => no apples-to-oranges).
from scripts.score_mc_kappa_pod import build_head_bases, _top_pcs

LAYER_BANDS = [("early_L0_19", 0, 19), ("mid_L20_33", 20, 33), ("late_L34_58", 34, 999)]


# ----------------------------------------------------------------------------
# Per-head decomposition (float64, pure geometry; no forward passes).
# ----------------------------------------------------------------------------
def decompose_theta_head(theta_vec, wk, wp, pcs=None, n_rand=2000, n_plane=300,
                         cond_max=1e4, gen=None):
    """Decompose unit(theta_vec) in span{wk, wp}; return metrics + matched nulls.

    theta_vec : fd['theta_mc'][ai,hi]  (applied offset, NOT unit; norm = theta_norm_raw)
    wk, wp    : W_know[:,0], W_pred[:,0] from build_head_bases (unit)
    pcs       : [d, K] top PCs of this head's cov_inf (PC1..PCK), for N2b/N3 nulls.
                pcs[:,0] should match wp (up to sign) when wpred_source == 'pc1'.
    """
    d = theta_vec.shape[0]
    tn = float(torch.linalg.norm(theta_vec.double()))
    if tn < 1e-9:
        return {"skip": "zero_theta"}
    th = theta_vec.double() / tn
    wk = wk.double() / torch.linalg.norm(wk.double())
    wp = wp.double() / torch.linalg.norm(wp.double())

    g = float(wk @ wp)
    cond_G = (1.0 + abs(g)) / max(1.0 - abs(g), 1e-12)
    degenerate = cond_G > cond_max
    cos_know = float(wk @ th)
    cos_pred = float(wp @ th)

    out = dict(g=g, cond_G=cond_G, degenerate_basis=degenerate,
               cos_know=cos_know, cos_pred=cos_pred, theta_norm_raw=tn)

    # VERIFY-FIX [BAJA]: bail BEFORE the QR on a rank-deficient plane (QR would
    # return a garbage column). Knowledge/plane numbers that don't need wp_perp
    # are still valid and reported.
    if degenerate:
        out.update(R2_span=float("nan"), R2_perp=float("nan"),
                   R2_know_pure=cos_know ** 2, R2_pred_pure=float("nan"),
                   R2_pred_axis=cos_pred ** 2, R2_know_resid=float("nan"),
                   cos_pred_perp=float("nan"), cos_know_perp=float("nan"),
                   c_know=float("nan"), c_pred=float("nan"), cross_term=float("nan"),
                   p_E_in=float("nan"), p_E_pred=float("nan"),
                   p_high_know=float("nan"), p_low_know=float("nan"),
                   p_plane_N2a=float("nan"), p_plane_N2b=float("nan"),
                   R2_pred_pc1=float("nan"), R2_pred_pc_ge2=float("nan"),
                   p_pc1_over_pcge2=float("nan"),
                   R2_span_random_mean=float("nan"), R2_axis_random_mean=float("nan"))
        return out

    # --- QR of the plane: orthonormal basis => obliquity-free projection -----
    B = torch.stack([wk, wp], dim=1)                  # [d,2]
    Q, _ = torch.linalg.qr(B, mode="reduced")         # [d,2] orthonormal span{wk,wp}
    proj = Q.t() @ th
    R2_span = float(proj @ proj)
    R2_perp = 1.0 - R2_span

    # --- asymmetric partition (wk 1D absorbs overlap) ------------------------
    R2_know_pure = cos_know ** 2
    wp_perp = wp - (wp @ wk) * wk
    wp_perp = wp_perp / torch.linalg.norm(wp_perp)
    cos_pred_perp = float(wp_perp @ th)
    R2_pred_pure = cos_pred_perp ** 2
    assert abs((R2_know_pure + R2_pred_pure) - R2_span) < 1e-5, \
        f"Pythagoras partition broke: {R2_know_pure}+{R2_pred_pure} != {R2_span}"

    # --- symmetric complement (wp 1D absorbs overlap) — VERIFY-FIX [ALTA-1] --
    R2_pred_axis = cos_pred ** 2
    wk_perp = wk - (wk @ wp) * wp
    wk_perp = wk_perp / torch.linalg.norm(wk_perp)
    cos_know_perp = float(wk_perp @ th)
    R2_know_resid = cos_know_perp ** 2

    # --- oblique coords + Gram-identity assert — VERIFY-FIX [MED] ------------
    G = torch.tensor([[1.0, g], [g, 1.0]], dtype=torch.float64)
    c = torch.linalg.solve(G, torch.tensor([cos_know, cos_pred], dtype=torch.float64))
    c_know, c_pred = float(c[0]), float(c[1])
    quad = float(c @ G @ c)                            # must equal R2_span
    assert abs(quad - R2_span) < 1e-5, \
        f"oblique Gram identity broke: c^T G c={quad} != R2_span={R2_span}"
    cross_term = 2.0 * g * c_know * c_pred

    # --- matched nulls -------------------------------------------------------
    if gen is None:
        gen = torch.Generator().manual_seed(0)
    R = torch.randn(n_rand, d, generator=gen, dtype=torch.float64)
    R = R / R.norm(dim=1, keepdim=True)
    Rspan_r = (R @ Q).pow(2).sum(1)
    p_E_in = float((Rspan_r >= R2_span).double().mean())
    Ekp_r = (R @ wk).pow(2)
    p_high_know = float((Ekp_r >= R2_know_pure).double().mean())
    p_low_know = float((Ekp_r <= R2_know_pure).double().mean())   # VERIFY-FIX [ALTA-2]
    Epp_r = (R @ wp_perp).pow(2)
    p_E_pred = float((Epp_r >= R2_pred_pure).double().mean())

    # N2a: theta fixed vs random 2-plane
    plane_a = []
    for _ in range(n_plane):
        M = torch.randn(d, 2, generator=gen, dtype=torch.float64)
        Qr, _ = torch.linalg.qr(M, mode="reduced")
        plane_a.append(float((Qr.t() @ th).pow(2).sum()))
    plane_a = torch.tensor(plane_a, dtype=torch.float64)
    p_plane_N2a = float((plane_a >= R2_span).double().mean())

    # N2b / N3: PC-conditioned nulls (separate "KAPPA plane" / "PC1 specifically"
    # from "any high-variance axis"). VERIFY-FIX [MED].
    p_plane_N2b = float("nan")
    R2_pred_pc1 = float("nan")
    R2_pred_pc_ge2 = float("nan")
    p_pc1_over_pcge2 = float("nan")
    if pcs is not None and pcs.shape[1] >= 2:
        pcs = pcs.double()
        pc1 = pcs[:, 0] / torch.linalg.norm(pcs[:, 0])
        # N2b: plane {PC1, random} -- does adding the high-var PC1 alone explain R2_span?
        plane_b = []
        for _ in range(n_plane):
            r = torch.randn(d, generator=gen, dtype=torch.float64)
            Mb = torch.stack([pc1, r], dim=1)
            Qb, _ = torch.linalg.qr(Mb, mode="reduced")
            plane_b.append(float((Qb.t() @ th).pow(2).sum()))
        plane_b = torch.tensor(plane_b, dtype=torch.float64)
        p_plane_N2b = float((plane_b >= R2_span).double().mean())
        # N3: energy on PC1 vs PC2..PCk (rank-matched). Only meaningful as the pred
        # null when wpred_source==pc1 (then wp ~ pc1). Reported regardless; the
        # assembler reads it only for the pc1 run.
        R2_pred_pc1 = float((pc1 @ th) ** 2)
        ge2 = (pcs[:, 1:].t() @ th).pow(2)             # [K-1]
        R2_pred_pc_ge2 = float(ge2.mean())
        # one-sided: fraction of PC>=2 axes whose energy >= PC1 energy
        p_pc1_over_pcge2 = float((ge2 >= R2_pred_pc1).double().mean())

    out.update(
        R2_span=R2_span, R2_perp=R2_perp,
        R2_know_pure=R2_know_pure, R2_pred_pure=R2_pred_pure,
        R2_pred_axis=R2_pred_axis, R2_know_resid=R2_know_resid,
        cos_pred_perp=cos_pred_perp, cos_know_perp=cos_know_perp,
        c_know=c_know, c_pred=c_pred, cross_term=cross_term,
        R2_span_random_mean=float(Rspan_r.mean()),   # ~2/d
        R2_axis_random_mean=float(Ekp_r.mean()),     # ~1/d
        p_E_in=p_E_in, p_E_pred=p_E_pred,
        p_high_know=p_high_know, p_low_know=p_low_know,
        p_plane_N2a=p_plane_N2a, p_plane_N2b=p_plane_N2b,
        R2_pred_pc1=R2_pred_pc1, R2_pred_pc_ge2=R2_pred_pc_ge2,
        p_pc1_over_pcge2=p_pc1_over_pcge2)
    return out


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------
def _summ(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "p5": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)),
            "min": float(x.min()), "max": float(x.max())}


def _bimodality_coef(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if x.size < 4:
        return None
    m = x.mean(); s = x.std()
    if s < 1e-12:
        return 0.0
    z = (x - m) / s
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())                      # non-excess
    if kurt < 1e-9:
        return None
    n = x.size
    # sample-corrected bimodality coefficient; > 0.555 suggests bimodality
    return float((skew ** 2 + 1) / (kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def _boot_ci(x, weights=None, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    x = x[ok]
    if x.size == 0:
        return [None, None]
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[ok]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        if w is None:
            means[b] = x[idx].mean()
        else:
            ww = w[idx]
            means[b] = np.average(x[idx], weights=ww) if ww.sum() > 0 else x[idx].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _wmean(vals, w):
    vals = np.asarray(vals, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(vals) & np.isfinite(w)
    if not ok.any() or w[ok].sum() <= 0:
        return None
    return float(np.average(vals[ok], weights=w[ok]))


def _bh_fdr_frac(pvals, q=0.05):
    """Fraction of heads significant at BH-FDR q over the given one-sided p-values."""
    p = np.asarray([v for v in pvals if v is not None and np.isfinite(v)], dtype=np.float64)
    if p.size == 0:
        return 0.0
    m = p.size
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * q
    passed = p[order] <= thresh
    k = np.where(passed)[0]
    n_sig = (k.max() + 1) if k.size else 0
    return float(n_sig / m)


# ----------------------------------------------------------------------------
def derive_adapter_pairs(fd):
    """The (layer,head) the adapter actually modulates: theta_mc nonzero. Derived
    from functional_directions.pt alone (no model / offsets blob needed)."""
    theta = fd["theta_mc"]                              # [n_cap, nH, D] array-space
    captured = [int(x) for x in fd["captured_indices"]]
    pairs = []
    for ai, li in enumerate(captured):
        for hi in range(theta.shape[1]):
            if float(torch.linalg.norm(theta[ai, hi].double())) > 1e-6:
                pairs.append((li, hi))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--func", default="runs/functional_directions.pt")
    ap.add_argument("--wpred-source", default="pc1", choices=["pc1", "vinf", "wknow"])
    ap.add_argument("--k-know", type=int, default=1)
    ap.add_argument("--k-pred", type=int, default=1)
    ap.add_argument("--ridge-frac", type=float, default=1e-2,
                    help="LDA/PC ridge frac (matches make_wknow_offset.py / KAPPA foil)")
    ap.add_argument("--n-rand", type=int, default=2000)
    ap.add_argument("--n-plane", type=int, default=300)
    ap.add_argument("--n3-pc", type=int, default=8, help="top-K PCs for the N2b/N3 nulls")
    ap.add_argument("--cond-max", type=float, default=1e4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print("=" * 74, flush=True)
    print(f"theta_mc decomposition in KAPPA {{W_know,W_pred}} | wpred={args.wpred_source} "
          f"k_know={args.k_know} k_pred={args.k_pred}", flush=True)
    print("  geometry-only, no model; reuses build_head_bases (same W_know/W_pred as GATE-2)",
          flush=True)
    print("=" * 74, flush=True)

    fd = torch.load(args.func, map_location="cpu", weights_only=False)
    pairs = derive_adapter_pairs(fd)
    only_layers = sorted({li for li, _ in pairs})
    print(f"  adapter heads (theta_mc nonzero): {len(pairs)} on {len(only_layers)} layers",
          flush=True)

    head_bases, diag = build_head_bases(
        pairs, fd, args.k_know, args.k_pred, args.wpred_source, args.ridge_frac,
        set(only_layers))
    print(f"  [build] {json.dumps(diag)}", flush=True)
    if diag["n_built"] == 0:
        raise SystemExit("FATAL: 0 heads built (check cov_inf / captured_indices).")

    cov_inf = fd["cov_inf"]
    gen = torch.Generator().manual_seed(args.seed)
    per_head = []
    n_built = sum(len(v) for v in head_bases.values())
    i = 0
    for li, items in sorted(head_bases.items()):
        for (hi, W_know, W_pred, theta) in items:
            i += 1
            cov = cov_inf.get((li, hi))
            pcs = _top_pcs(cov, args.n3_pc, args.ridge_frac) if cov is not None else None
            r = decompose_theta_head(theta, W_know[:, 0], W_pred[:, 0], pcs=pcs,
                                     n_rand=args.n_rand, n_plane=args.n_plane,
                                     cond_max=args.cond_max, gen=gen)
            if "skip" in r:
                continue
            r["li"] = int(li); r["hi"] = int(hi)
            per_head.append(r)
            if i % 10 == 0 or i == n_built:
                print(f"  [{i}/{n_built}] decomposed (L{li} H{hi} "
                      f"R2_span={r['R2_span']:.4f} R2_perp={r['R2_perp']:.4f})", flush=True)

    if not per_head:
        # wknow self-test: all degenerate is the EXPECTED outcome.
        all_degen = (args.wpred_source == "wknow")
        print(f"  no decomposable heads (wpred={args.wpred_source}). "
              f"all_degenerate_expected={all_degen}", flush=True)

    # --- aggregation -------------------------------------------------------
    def col(key):
        return [h.get(key) for h in per_head]
    w_energy = [h["theta_norm_raw"] ** 2 for h in per_head]

    metrics = ["R2_span", "R2_perp", "R2_know_pure", "R2_pred_pure",
               "R2_pred_axis", "R2_know_resid", "cos_know", "cos_pred",
               "cos_pred_perp", "c_know", "c_pred", "cross_term"]
    unweighted = {m: _summ(col(m)) for m in metrics}
    for m in ("R2_know_pure", "R2_pred_pure", "R2_span"):
        unweighted[m]["bimodality_coef"] = _bimodality_coef(col(m))

    energy_weighted = {
        f"{m}_wmean": _wmean(col(m), w_energy)
        for m in ("R2_span", "R2_perp", "R2_know_pure", "R2_pred_pure")}

    by_band = {}
    for name, lo, hi in LAYER_BANDS:
        sub = [h for h in per_head if lo <= h["li"] <= hi]
        by_band[name] = {"n": len(sub),
                         "R2_span_mean": (float(np.mean([h["R2_span"] for h in sub]))
                                          if sub else None),
                         "R2_pred_pure_mean": (float(np.nanmean([h["R2_pred_pure"] for h in sub]))
                                               if sub else None),
                         "R2_know_pure_mean": (float(np.mean([h["R2_know_pure"] for h in sub]))
                                               if sub else None)}

    boot = {
        "R2_span_mean": _boot_ci(col("R2_span"), seed=args.seed),
        "R2_span_mean_energyweighted": _boot_ci(col("R2_span"), weights=w_energy, seed=args.seed),
        "R2_pred_pure_mean": _boot_ci(col("R2_pred_pure"), seed=args.seed),
        "R2_know_pure_mean": _boot_ci(col("R2_know_pure"), seed=args.seed)}

    def _hist(key, bins):
        x = np.asarray([v for v in col(key) if v is not None and np.isfinite(v)])
        if x.size == 0:
            return {"bins": list(bins), "counts": []}
        cnt, edges = np.histogram(x, bins=bins)
        return {"bins": [float(e) for e in edges], "counts": [int(c) for c in cnt]}
    hist_bins = np.linspace(0.0, 0.2, 21)
    histograms = {k: _hist(k, hist_bins) for k in ("R2_span", "R2_pred_pure", "R2_know_pure")}

    # nulls / significance
    n2a = [h["p_plane_N2a"] for h in per_head]
    n2b = [h["p_plane_N2b"] for h in per_head]
    n2 = {"frac_heads_R2span_over_N2a_random_plane_p05":
          float(np.mean([1.0 if (p is not None and np.isfinite(p) and p < 0.05) else 0.0
                         for p in n2a])) if n2a else 0.0,
          "frac_heads_R2span_over_N2b_pc1random_plane_p05":
          float(np.mean([1.0 if (p is not None and np.isfinite(p) and p < 0.05) else 0.0
                         for p in n2b if (p is not None and np.isfinite(p))])) if any(
              p is not None and np.isfinite(p) for p in n2b) else None,
          "p_plane_N2a_summary": _summ(n2a), "p_plane_N2b_summary": _summ(n2b)}

    n3 = {"R2_pred_pc1_mean": (float(np.nanmean(col("R2_pred_pc1")))
                               if any(np.isfinite(v) for v in col("R2_pred_pc1")) else None),
          "R2_pred_pc_ge2_mean": (float(np.nanmean(col("R2_pred_pc_ge2")))
                                  if any(np.isfinite(v) for v in col("R2_pred_pc_ge2")) else None),
          "frac_heads_pc1_over_pcge2_null_p05":
          float(np.mean([1.0 if (h.get("p_pc1_over_pcge2") is not None
                                 and np.isfinite(h["p_pc1_over_pcge2"])
                                 and h["p_pc1_over_pcge2"] < 0.05) else 0.0
                         for h in per_head])) if per_head else 0.0,
          "note": ("R2_pred_pure(pc1) is interpretable ONLY against R2_pred_pc_ge2 "
                   "(VERIFY-FIX MED #4): if pc1 ~ pc>=2, the pred energy is a generic "
                   "high-variance-axis artifact, not a KAPPA-prediction alignment.")}

    significance = {
        "frac_heads_E_pred_over_null_fdr05": _bh_fdr_frac(col("p_E_pred")),
        "frac_heads_E_low_know_fdr05": _bh_fdr_frac(col("p_low_know")),  # ⊥W_know (C1)
        "n_degenerate_basis": int(sum(1 for h in per_head if h.get("degenerate_basis"))),
        "n_heads_decomposed": len(per_head)}

    # --- verdict (battery-spec head-2 decision rule) -----------------------
    floor_2d = 2.0 / int(fd["head_dim"])               # E[R2_span] random plane ~ 2/d
    r2span_w = energy_weighted["R2_span_wmean"]
    pred_survives_n3 = (n3["R2_pred_pc1_mean"] is not None and n3["R2_pred_pc_ge2_mean"] is not None
                        and n3["R2_pred_pc1_mean"] > 2.0 * n3["R2_pred_pc_ge2_mean"]
                        and n3["frac_heads_pc1_over_pcge2_null_p05"] > 0.5)
    know_energy_high = (energy_weighted["R2_know_pure_wmean"] is not None
                        and energy_weighted["R2_know_pure_wmean"] > 5.0 * (1.0 / int(fd["head_dim"]))
                        and float(np.nanmean(col("R2_know_resid"))) > 5.0 * (1.0 / int(fd["head_dim"])))
    if args.wpred_source == "wknow":
        outcome = "SELFTEST"
        wording = ("self-test: wpred==wknow => degenerate basis expected on every head "
                   f"(n_degenerate={significance['n_degenerate_basis']}/{len(pairs)} pairs).")
    elif know_energy_high:
        outcome = "D"
        wording = ("ALARM: theta_mc carries knowledge energy in BOTH partitions — "
                   "contradicts off-axis. Audit cond_G / wk-orthogonalized re-run / cap_index.")
    elif r2span_w is not None and r2span_w <= 3.0 * floor_2d:
        outcome = "A"
        wording = ("HEADLINE (C1+C3 by exclusion): the theta_mc DIRECTION is orthogonal to "
                   "W_know AND lies outside the {W_know,W_pred} plane of KAPPA (energy ~ random-"
                   "plane floor). Wording: where the offset DIRECTION lives, not what it does.")
    elif pred_survives_n3:
        outcome = "B"
        wording = ("EXPLORATORY (C4, inverse-of-KAPPA): pred-coord energy survives the PC>=2 "
                   "null. CAVEAT: W_pred is the reconstructed PC1, not the paper's greedy-option "
                   "probe; confirm before claiming 'rewrites prediction'.")
    else:
        outcome = "C"
        wording = ("C4 falls (pred energy proxy-dependent / fails N3); C3 (exclusion) stands. "
                   "Report R2_span vs floor and the pc1-vs-vinf divergence.")

    payload = {
        "config": {"func": str(args.func), "wpred_source": args.wpred_source,
                   "k_know": args.k_know, "k_pred": args.k_pred,
                   "ridge_frac": args.ridge_frac, "n_rand": args.n_rand,
                   "n_plane": args.n_plane, "n3_pc": args.n3_pc, "cond_max": args.cond_max,
                   "seed": args.seed, "head_dim": int(fd["head_dim"]),
                   "n_adapter_pairs": len(pairs), "n_heads_decomposed": len(per_head),
                   "head_set": "adapter (theta_mc nonzero), cov_inf-eligible, "
                               "unfiltered-by-model (geometry exact)",
                   "space": "per-head o_proj-input (d=256); forward-filter same-space",
                   "measures": "WHERE the offset DIRECTION lives, NOT what the offset DOES "
                               "(effect side W_o.theta is out of scope, follow-up)"},
        "null_baselines": {"R2_span_random_plane_floor_2overD": floor_2d,
                           "R2_axis_random_floor_1overD": 1.0 / int(fd["head_dim"]),
                           "d": int(fd["head_dim"])},
        "build_diagnostics": diag,
        "aggregate": {"unweighted": unweighted, "offset_energy_weighted": energy_weighted,
                      "by_layer_band": by_band, "bootstrap_ci_percentile": boot,
                      "histograms": histograms},
        "nulls": {"N2_random_plane": n2, "N3_pc_rank_matched": n3,
                  "significance_fdr": significance},
        "verdict": {"outcome": outcome, "wording": wording,
                    "R2_span_energyweighted_mean": r2span_w,
                    "R2_span_random_plane_floor": floor_2d,
                    "cos_wknow_theta_mean_from_build": diag.get("cos_wknow_theta_mean"),
                    "cos_wpred_wknow_mean_from_build": diag.get("cos_wpred_wknow_mean")},
        "per_head": per_head,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 74, flush=True)
    print(f"  VERDICT [{outcome}] {wording}", flush=True)
    print(f"  R2_span (energy-weighted mean) = {r2span_w}  vs random-plane floor {floor_2d:.5f}",
          flush=True)
    print(f"  cos(W_know,theta) mean = {diag.get('cos_wknow_theta_mean')}  "
          f"cos(W_pred,W_know) mean = {diag.get('cos_wpred_wknow_mean')}", flush=True)
    print(f"  saved: {args.out}", flush=True)
    print("  next: run --wpred-source vinf (cross-proxy M5) + --wpred-source wknow (self-test); "
          "assemble head-2 verdict alongside GATE-2 head-1 (Δ vs KAPPA).", flush=True)


if __name__ == "__main__":
    main()
