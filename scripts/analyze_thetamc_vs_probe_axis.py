"""L10 — theta_mc vs the probe's read-axis (audit 2026-07-02 §6-P1, §9-R7).

The geometric blind spot: the in-space probe decodes correctness at AUC 0.955, but the
adapter's WRITE direction (theta_mc) was never decomposed against the probe's DECISION
direction (the optimal in-space READER of correctness). If theta_mc is ~orthogonal to the
probe axis too, the off-axis claim is blinded against "your on-axis references (v_inf/
W_know) were weak references".

This REUSES the validated probe-direction machinery
(analyze_probe_direction_geometry.fit_probe_dir) — not a re-implementation — and adds a
theta_mc-specific random null band, a sanity that cos(probe, v_inf) reproduces the paper's
pooled +0.209, and the pre-registered gate. Cross-checks against the existing
probe_direction_geometry.json.

Gate (plan L10-Step3):
  |pooled cos(theta_mc, probe)| INSIDE the null band -> UPGRADE: the adapter writes
    off-axis even w.r.t. the probe's decision direction.
  |cos| OUTSIDE -> refine: theta_mc has a component on the read-axis; report it honestly.

CPU-local, EUR0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_probe_direction_geometry import (  # reuse, do NOT copy
    fit_probe_dir, cos_raw, cos_whit, whiten_M, dist, MC, FUNC_CANDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, default=Path("runs/vinf_causal/coldmc_perhead_arc_challenge.pt"))
    ap.add_argument("--C", type=float, default=0.03)
    ap.add_argument("--pca", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--sanity-vinf-pooled", type=float, default=0.209,
                    help="expected pooled cos(probe, v_inf); reproduce within tol or BUG")
    ap.add_argument("--sanity-tol", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=Path("runs/vinf_causal/thetamc_vs_probe_axis.json"))
    args = ap.parse_args()

    print(f"[load] {args.acts}", flush=True)
    st = torch.load(args.acts, weights_only=False, map_location="cpu")
    A = st["acts_adapter"]
    nP, nC, hd = A.shape
    X = A.reshape(nP, nC * hd).numpy().astype(np.float64)
    y = np.array([m["is_gold"] for m in st["prompt_meta"]], dtype=np.int64)
    cells = [tuple(int(z) for z in c) for c in st["adapter_cells_layer_head"]]
    print(f"[load] cells={nC} rows={nP} head_dim={hd}", flush=True)

    mc = torch.load(MC, weights_only=False, map_location="cpu")
    mc_pairs = [tuple(int(z) for z in p) for p in mc["layer_head_pairs"]]
    assert mc_pairs == cells, "cell order mismatch acts vs offsets_mc"
    theta = mc["theta"].to(torch.float64).numpy()
    theta_unit = theta / np.linalg.norm(theta, axis=1, keepdims=True)

    func = next((p for p in FUNC_CANDS if p.exists()), None)
    if func is None:
        raise SystemExit("functional_directions.pt not found")
    fd = torch.load(func, weights_only=False, map_location="cpu")
    v_inf = fd["v_inf"].to(torch.float64).numpy()
    captured = [int(z) for z in fd["captured_indices"]]
    cap = {L: ai for ai, L in enumerate(captured)}

    print(f"[fit] probe direction on all rows (C={args.C}, pca={args.pca}) — "
          f"direction-extraction, NOT held-out eval", flush=True)
    w_raw, sc, pca, clf = fit_probe_dir(X, y, args.C, args.pca, args.seed)
    P = w_raw.reshape(nC, hd)

    v_unit = np.zeros((nC, hd))
    for i, (L, h) in enumerate(cells):
        v = v_inf[cap[L], h]
        v_unit[i] = v / np.linalg.norm(v)

    cos_pt, cos_pv = [], []      # |cos(probe, theta)| ; signed cos(probe, v_inf)
    for i in range(nC):
        cos_pt.append(abs(cos_raw(P[i], theta_unit[i])))
        cos_pv.append(cos_raw(P[i], v_unit[i]))
    Pp, tp, vp = P.reshape(-1), theta_unit.reshape(-1), v_unit.reshape(-1)
    pooled_theta = abs(cos_raw(Pp, tp))
    pooled_vinf = cos_raw(Pp, vp)

    # theta_mc-specific random null band: mean over heads of |cos(random, theta_unit)|
    print(f"[null] {args.n_null} random-direction draws vs theta_mc ...", flush=True)
    rng = np.random.default_rng(args.seed)
    null_means = []
    for t in range(args.n_null):
        r = rng.standard_normal((nC, hd))
        null_means.append(float(np.mean([abs(cos_raw(r[i], theta_unit[i])) for i in range(nC)])))
        if (t + 1) % 250 == 0:
            print(f"    [null {t+1}/{args.n_null}]", flush=True)
    null_lo, null_hi = float(np.percentile(null_means, 2.5)), float(np.percentile(null_means, 97.5))
    null_theory = float(1.0 / np.sqrt(hd))

    # single-draw null band (for the POOLED scalar comparison, a single 12288-d cos)
    pooled_null = [abs(cos_raw(rng.standard_normal(nC * hd), tp)) for _ in range(args.n_null)]
    pooled_null_hi = float(np.percentile(pooled_null, 97.5))

    theta_mean_abs = float(np.mean(cos_pt))
    vinf_mean = float(np.mean(cos_pv))

    # sanity: probe direction must reproduce the paper's cos(probe, v_inf)
    sanity_ok = abs(pooled_vinf - args.sanity_vinf_pooled) <= args.sanity_tol
    print(f"[sanity] pooled cos(probe, v_inf) = {pooled_vinf:+.4f} "
          f"(expected ~{args.sanity_vinf_pooled}; ok={sanity_ok})", flush=True)
    if not sanity_ok:
        raise SystemExit(f"SANITY FAIL: pooled cos(probe,v_inf)={pooled_vinf:.4f} does not "
                         f"reproduce ~{args.sanity_vinf_pooled} -> direction extraction BUG, stop.")

    # GATE
    pooled_inside = pooled_theta <= pooled_null_hi
    perhead_at_or_below = theta_mean_abs <= null_hi
    if pooled_inside:
        gate = ("UPGRADE: |pooled cos(theta_mc, probe)|={:.4f} is INSIDE the null band "
                "(<= p97.5 {:.4f}); the adapter writes off-axis even w.r.t. the probe's "
                "decision direction. Per-head |cos| mean {:.4f} vs per-head-mean null p97.5 "
                "{:.4f} ({}). Sanity cos(probe,v_inf) pooled {:.4f} reproduces the paper."
                ).format(pooled_theta, pooled_null_hi, theta_mean_abs, null_hi,
                         "at/below null" if perhead_at_or_below else "slightly above null",
                         pooled_vinf)
    else:
        gate = ("REFINE: |pooled cos(theta_mc, probe)|={:.4f} is OUTSIDE the null band "
                "(> p97.5 {:.4f}); theta_mc has a component on the read-axis — report as "
                "'off-axis to v_inf/W_know; partially aligned to the probe read-axis "
                "(cos={:.4f})'.").format(pooled_theta, pooled_null_hi, pooled_theta)
    print(f"[gate] {gate}", flush=True)

    out = {
        "task": st["task"], "n_cells": nC, "n_rows": nP, "head_dim": hd,
        "C": args.C, "pca": args.pca, "seed": args.seed, "n_null": args.n_null,
        "cos_probe_theta_abs_perhead": dist(cos_pt),
        "cos_probe_theta_abs_pooled": pooled_theta,
        "cos_probe_vinf_signed_perhead": dist(cos_pv),
        "cos_probe_vinf_pooled": pooled_vinf,
        "null_band_theory_1_over_sqrt_d": null_theory,
        "null_perhead_mean_ci95": [null_lo, null_hi],
        "null_pooled_p97.5": pooled_null_hi,
        "sanity_cos_probe_vinf_pooled_ok": bool(sanity_ok),
        "gate_verdict": gate,
        "note": ("Independent reproduction of the theta-vs-probe leg of "
                 "probe_direction_geometry.json; PC-baseline caveat (cos(PC1,v_inf) 0.55 > "
                 "cos(probe,v_inf) 0.22) lives in that file and concerns the READ leg, not this "
                 "WRITE-vs-read result."),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)

    # cross-check vs the existing geometry artifact
    geo = Path("runs/vinf_causal/probe_direction_geometry.json")
    if geo.exists():
        g = json.load(open(geo))
        print(f"[xcheck] existing pooled_raw_theta_abs={g['pooled_raw_theta_abs']:.5f} vs "
              f"mine {pooled_theta:.5f} | existing perhead mean "
              f"{g['raw_cos_probe_theta_abs']['mean']:.5f} vs mine {theta_mean_abs:.5f} | "
              f"existing pooled_raw_vinf={g['pooled_raw_vinf']:.5f} vs mine {pooled_vinf:.5f}",
              flush=True)


if __name__ == "__main__":
    main()
