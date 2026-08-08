"""Decompose the contrastive mc adapter's per-head offset into the component
PARALLEL vs ORTHOGONAL to each candidate on-axis functional direction. €0-local.

For each adapter head i = (layer, head), the applied offset is
    off = alpha[i] * theta[i]    (head_dim space)
We project off onto unit(v) for v in {v_inf, v_ret, W_know} and report, per head
and aggregated, the ON-AXIS ENERGY FRACTION  proj^2 / ||off||^2 . The aggregate
energy-weighted budget  (sum_i proj_i^2) / (sum_i ||off_i||^2)  is the headline:
"how much of the trained offset's energy a given functional direction captures".
If even W_know (the strongest buildable on-axis reference, Fisher-LDA) captures a
tiny fraction, the adapter's push is OFF-AXIS robustly (consistent with W_know
recovering only 6.4% of the ARC lift).

  Block A: raw head_dim euclidean decomposition (all adapter heads on the E2 grid).
  Block B: whitened (Mahalanobis under cov_inf, eligible heads) cos(off, v_inf),
           reproducing the canonical cos_w(theta_mc, v_inf) ~ -0.003.

NOTE ON SPACE (forward filter: same-space): all directions live in the per-head
o_proj-input (head_dim) space, the space where theta_mc/v_inf/W_know were defined.
This is a GEOMETRIC budget (norm), NOT the readout effect — the scoring/gen effect
of off_par vs off_perp is a separate pod measurement (geometry->behaviour is
non-linear through o_proj + the cold-MC readout).

  MSAP_ROOT=. python scripts/analyze_offset_parperp_decomposition.py
"""
import json
import os

import numpy as np
import torch

ROOT = os.environ.get("MSAP_ROOT", ".")
MC = os.environ.get("MC_OFFSETS", os.path.join(ROOT, "runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt"))
FUNC = os.environ.get("FUNC_DIRS", os.path.join(ROOT, "runs/functional_directions.pt"))
WK = os.environ.get("WKNOW_OFFSET", os.path.join(ROOT, "runs/vinf_causal/mc_wknow_offset.pt"))
OUT = os.environ.get("PARPERP_OUT", os.path.join(ROOT, "runs/derisk/offset_parperp_decomposition.json"))
SHRINK = float(os.environ.get("WKNOW_SHRINKAGE", "1e-2"))  # match make_wknow_offset


def unit(x):
    n = torch.linalg.norm(x)
    return x / n if n > 1e-12 else x


def summ(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), max=float(a.max()), min=float(a.min()))


def main():
    print("=" * 70)
    print("offset par/perp decomposition vs {v_inf, v_ret, W_know} — raw + whitened")
    print("=" * 70)
    mc = torch.load(MC, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in p) for p in mc["layer_head_pairs"]]
    alpha = mc["alpha"].to(torch.float64)
    theta = mc["theta"].to(torch.float64)

    fd = torch.load(FUNC, weights_only=False, map_location="cpu")
    v_inf = fd["v_inf"].to(torch.float64)
    v_ret = fd["v_ret"].to(torch.float64)
    captured = [int(x) for x in fd["captured_indices"]]
    cap = {li: ai for ai, li in enumerate(captured)}
    cov_inf = fd["cov_inf"]
    head_dim = int(fd["head_dim"])

    wk = torch.load(WK, weights_only=False, map_location="cpu")
    wk_pairs = [tuple(int(x) for x in p) for p in wk["layer_head_pairs"]]
    assert wk_pairs == pairs, "W_know pair order != mc pair order — cannot join"
    wk_theta = wk["theta"].to(torch.float64)  # W_know unit direction per head

    print(f"  adapter heads: {len(pairs)}   head_dim: {head_dim}")
    print(f"  captured E2 layers: {len(captured)}   cov_inf cells: {len(cov_inf)}")

    DIRS = ["v_inf", "v_ret", "W_know"]
    agg = {k: {"proj2": 0.0, "norm2": 0.0, "fracs": [], "coss": []} for k in DIRS}
    per_head = []
    offs = []          # per-head offset vectors (for the random-direction null)
    n_used = n_skip = 0
    for i, (li, hi) in enumerate(pairs):
        if li not in cap:
            n_skip += 1
            continue
        ai = cap[li]
        off = alpha[i] * theta[i]
        n2 = float(off @ off)
        n = n2 ** 0.5
        offs.append(off)
        rec = {"layer": li, "head": hi, "norm_off": n}
        cand = {"v_inf": v_inf[ai, hi], "v_ret": v_ret[ai, hi], "W_know": wk_theta[i]}
        for k, v in cand.items():
            vh = unit(v)
            proj = float(off @ vh)
            frac = (proj * proj / n2) if n2 > 0 else 0.0
            cos = (proj / n) if n > 0 else 0.0
            rec[f"frac_{k}"] = frac
            rec[f"cos_{k}"] = cos
            agg[k]["proj2"] += proj * proj
            agg[k]["norm2"] += n2
            agg[k]["fracs"].append(frac)
            agg[k]["coss"].append(cos)
        per_head.append(rec)
        n_used += 1

    # ---- random-direction null: is the on-axis budget above what a RANDOM
    # per-head direction captures? E[energy budget] ~ 1/head_dim for random dirs.
    # Null = distribution of the aggregate energy-weighted budget over K random
    # per-head direction sets. Reports z-score + percentile of each observed dir.
    K_NULL = int(os.environ.get("N_NULL", "2000"))
    torch.manual_seed(0)
    O = torch.stack(offs)                       # [n_used, head_dim]
    norm2_vec = (O * O).sum(1)                   # [n_used]
    tot_norm2 = float(norm2_vec.sum())
    null_budgets = np.empty(K_NULL, dtype=float)
    for k in range(K_NULL):
        R = torch.randn(O.shape, dtype=torch.float64)
        R = R / torch.linalg.norm(R, dim=1, keepdim=True)
        proj = (O * R).sum(1)                     # [n_used]
        null_budgets[k] = float((proj * proj).sum() / tot_norm2)
    null_mean = float(null_budgets.mean())
    null_std = float(null_budgets.std())
    null_p95 = float(np.percentile(null_budgets, 95))
    null_p99 = float(np.percentile(null_budgets, 99))
    null_info = {"K": K_NULL, "expected_1_over_d": 1.0 / head_dim,
                 "mean": null_mean, "std": null_std, "p95": null_p95, "p99": null_p99}

    block_a = {}
    for k in DIRS:
        e = agg[k]
        coss = np.array(e["coss"])
        budget = e["proj2"] / e["norm2"]
        z = (budget - null_mean) / null_std if null_std > 0 else float("nan")
        pctile = float((null_budgets < budget).mean() * 100.0)
        block_a[k] = {
            "on_axis_energy_budget": budget,  # energy-weighted aggregate
            "vs_random_null": {"ratio_to_null_mean": budget / null_mean,
                               "z_score": z, "percentile_in_null": pctile,
                               "above_null_p95": bool(budget > null_p95)},
            "per_head_frac_on_axis": summ(e["fracs"]),
            "per_head_cos": dict(mean=float(coss.mean()), mean_abs=float(np.abs(coss).mean()),
                                 median=float(np.median(coss)), min=float(coss.min()),
                                 max=float(coss.max())),
        }

    # ---- Block B: whitened (Mahalanobis under cov_inf) cos(off, v_inf) ----
    def maha_cos(off, v, cov):
        ridge = SHRINK * torch.diag(cov).mean()
        cr = cov + ridge * torch.eye(head_dim, dtype=torch.float64)
        pinv_v = torch.linalg.solve(cr, v)
        pinv_off = torch.linalg.solve(cr, off)
        ip = float(off @ pinv_v)
        no = float(off @ pinv_off)
        nv = float(v @ pinv_v)
        return ip / ((no * nv) ** 0.5) if (no > 0 and nv > 0) else 0.0

    wcos = []
    for i, (li, hi) in enumerate(pairs):
        if li not in cap:
            continue
        cov = cov_inf.get((li, hi))
        if cov is None:
            continue
        ai = cap[li]
        off = alpha[i] * theta[i]
        wcos.append(maha_cos(off, v_inf[ai, hi], cov.to(torch.float64)))
    wcos = np.array(wcos) if wcos else np.array([np.nan])
    block_b = {
        "v_inf_whitened": {
            "n_elig": int(np.isfinite(wcos).sum()),
            "cos_whitened": dict(mean=float(np.nanmean(wcos)), mean_abs=float(np.nanmean(np.abs(wcos))),
                                 median=float(np.nanmedian(wcos))),
            "frac_on_axis_whitened_mean": float(np.nanmean(wcos ** 2)),
            "note": "reproduces the canonical cos_w(theta_mc, v_inf) ~ -0.003 as consistency check",
        }
    }

    res = {
        "schema": "offset-parperp-1.0",
        "model": "gemma-4-31b-it",
        "adapter": "v7-mc (contrastive correctness)",
        "n_heads_used": n_used, "n_heads_skipped_off_grid": n_skip,
        "shrinkage_frac": SHRINK,
        "random_direction_null": null_info,
        "block_a_raw": block_a,
        "block_b_whitened": block_b,
        "per_head": per_head,
        "read": ("on_axis_energy_budget = fraction of the adapter's TOTAL offset energy "
                 "(summed over heads) captured by that on-axis direction. Small for ALL "
                 "three (incl. W_know) => off-axis robust. This is GEOMETRY (norm), not "
                 "the readout effect — off_par/off_perp scoring+gen effect is a pod measure."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"\n  heads used: {n_used}   skipped(off-grid): {n_skip}")
    print(f"\n  random-dir null: mean {null_mean*100:.3f}%  (1/d={1.0/head_dim*100:.3f}%)  "
          f"p95 {null_p95*100:.3f}%  p99 {null_p99*100:.3f}%  (K={K_NULL})")
    print("\n=== Block A — raw head_dim decomposition (vs random-direction null) ===")
    print(f"  {'direction':10s} {'budget':>8s} {'xNull':>6s} {'z':>7s} {'pctile':>7s} "
          f"{'>p95?':>6s} {'cos_abs':>8s}")
    for k in DIRS:
        a = block_a[k]
        r = a["vs_random_null"]
        print(f"  {k:10s} {a['on_axis_energy_budget']*100:7.2f}% {r['ratio_to_null_mean']:5.2f}x "
              f"{r['z_score']:+7.2f} {r['percentile_in_null']:6.1f}% "
              f"{str(r['above_null_p95']):>6s} {a['per_head_cos']['mean_abs']:8.4f}")
    print("\n=== Block B — whitened cos(off, v_inf), eligible heads ===")
    b = block_b["v_inf_whitened"]
    print(f"  n_elig={b['n_elig']}  cos_w mean {b['cos_whitened']['mean']:+.4f}  "
          f"mean_abs {b['cos_whitened']['mean_abs']:.4f}  median {b['cos_whitened']['median']:+.4f}")
    print(f"  (canonical cos_w(theta_mc, v_inf) ~ -0.003 — consistency check)")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
