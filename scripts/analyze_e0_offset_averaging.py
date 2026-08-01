"""E0 — does averaging independent adapters filter the off-axis component?

Pre-registration: docs/superpowers/plans/2026-08-01-e0-averaging-as-offaxis-filter-prereg.md
The kill-rules below were fixed BEFORE any contrast was computed.

WHAT THIS IS (EUR0, CPU-local, no model load, no forward passes)
===============================================================
The hypothesis under test ("averaging-as-a-filter"): the off-axis component of a
contrastive adapter is IDIOSYNCRATIC (seed-dependent) and the on-axis component is
SHARED, so averaging K independent replicas cancels the idiosyncratic part ~1/sqrt(K)
and leaves a direction with a HIGHER on-axis fraction than any part.

Geometry only. This measures WHERE the offset direction lives, never what it does --
same scope caveat as decompose_theta_mc_kappa_subspaces.py. E0 can only KILL the
idea; confirming it requires the pod.

W_know is built by build_head_bases() imported from score_mc_kappa_pod.py -- the SAME
estimator that produced the GATE-2 foil. No second estimator is introduced.

Per head h the vector the inference hook actually adds is  e_h = alpha_h * theta_h.

  f_know(e) = (w_hat_k . e_hat)^2        on-axis energy of the DIRECTION
  M         = mean_i e_i                 (element-wise over replicas, same heads)
  r_norm    = ||M|| / mean_i ||e_i||     cancellation factor
  Delta     = f_know(M) - mean_i f_know(e_i)

KILL-RULES (pre-registered)
  KR-A  power:       bootstrap CI95 of (Delta_S - Delta_N) must exclude 0.
  KR-B  materiality: median_h f_know(M_S) must reach the random floor 1/d.
  KR-C  diagnostic:  r_norm / cos_rep explain WHY (does not kill).

Usage (local WSL, ~1 min):
  python scripts/analyze_e0_offset_averaging.py \
      --func runs/functional_directions.pt \
      --out runs/e0_averaging/e0_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.score_mc_kappa_pod import build_head_bases  # noqa: E402

# --- the material, per the pre-registration -------------------------------------
SET_S = [  # replicas: same 48 heads, same corpus (tqa+arc), SEED is the only difference
    "runs/e1_plain_ce/offsets_seed0.final.pt",
    "runs/e1_plain_ce/offsets_seed1.final.pt",
    "runs/e1_plain_ce/offsets_seed2.final.pt",
]
SET_N = [  # null: what averaging three arbitrary things buys on its own
    "runs/derisk/offsets_random_s0.pt",
    "runs/derisk/offsets_random_s1.pt",
    "runs/derisk/offsets_random_s2.pt",
]
REFS = {
    "theta_mc": "runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt",
    "wknow_SELFTEST": "runs/vinf_causal/mc_wknow_offset.pt",
    "vinf": "runs/vinf_causal/mc_vinf_offset.pt",
    "off_par": "runs/vinf_causal/mc_offpar_offset.pt",
    "off_perp": "runs/vinf_causal/mc_offperp_offset.pt",
    "anti": "runs/e7_inverse/offsets_anti.final.pt",
    "p9_lam0": "runs/p9_onaxis/offsets_lam0.final.pt",
    "p9_lam5": "runs/p9_onaxis/offsets_lam5.final.pt",
    "p9_lam15": "runs/p9_onaxis/offsets_lam15.final.pt",
    "p9_lam50": "runs/p9_onaxis/offsets_lam50.final.pt",
}
SET_D = {  # exploratory: own head sets, NOT gated
    "coding": "runs/phase3_adapters/coding/offsets_correctness.final.pt",
    "factual": "runs/phase3_adapters/factual/offsets_correctness.final.pt",
    "math_proof": "runs/phase3_adapters/math_proof/offsets_correctness.final.pt",
    "socialiqa": "runs/phase3_adapters/socialiqa/offsets_correctness.final.pt",
    "kr_sciq": "runs/phase3_adapters/kr_sciq/offsets_correctness_k24_diagnostic.final.pt",
}


def load_effect(path):
    """-> (pairs, E) where E[j] = alpha_j * theta_j is what the hook adds at head j."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    pairs = [tuple(int(x) for x in p) for p in d["layer_head_pairs"]]
    E = (d["alpha"].float().unsqueeze(1) * d["theta"].float())
    return pairs, E


def unit(v, eps=1e-12):
    n = float(torch.linalg.norm(v))
    return (v / n) if n > eps else None


def f_know(e, wk):
    """On-axis energy of the DIRECTION of e: (w_hat . e_hat)^2 in [0, 1]."""
    eh = unit(e)
    if eh is None:
        return float("nan")
    return float((wk @ eh) ** 2)


def boot_ci(x, n_boot, rng, stat=np.mean, alpha=0.05):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    reps = stat(x[idx], axis=1)
    return float(np.percentile(reps, 100 * alpha / 2)), float(
        np.percentile(reps, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="runs/functional_directions.pt")
    ap.add_argument("--ridge-frac", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/e0_averaging/e0_result.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[1/6] loading functional_directions (~179 MB) ...", flush=True)
    fd = torch.load(args.func, map_location="cpu", weights_only=False)
    head_dim = int(fd.get("head_dim", 256))
    floor = 1.0 / head_dim
    print(f"      head_dim={head_dim}  random-direction floor 1/d = {floor:.6f}",
          flush=True)

    print("[2/6] loading the 48-head reference geometry ...", flush=True)
    ref_pairs, _ = load_effect(REFS["theta_mc"])
    print(f"      {len(ref_pairs)} (layer, head) pairs", flush=True)

    print("[3/6] building W_know per head (Fisher-LDA, same estimator as GATE-2) ...",
          flush=True)
    head_bases, diag = build_head_bases(
        ref_pairs, fd, k_know=1, k_pred=1, wpred_source="vinf",
        ridge_frac=args.ridge_frac, only_layers=None)
    WK = {}
    for li, entries in head_bases.items():
        for (hi, W_know, _W_pred, _th) in entries:
            WK[(li, hi)] = W_know[:, 0].float()
    print(f"      built {len(WK)} / {len(ref_pairs)} heads "
          f"(n_built={diag.get('n_built')}, skips: layer={diag.get('n_skip_layer_not_in_grid')}, "
          f"cov={diag.get('n_skip_no_cov')}, null={diag.get('n_skip_null_dir')})", flush=True)
    heads = [p for p in ref_pairs if p in WK]
    if len(heads) < len(ref_pairs):
        print(f"      WARNING: {len(ref_pairs) - len(heads)} heads without W_know, dropped",
              flush=True)

    def per_head_effects(paths):
        """-> [n_rep, n_heads] list of tensors aligned to `heads`."""
        out = []
        for p in paths:
            pr, E = load_effect(p)
            idx = {q: j for j, q in enumerate(pr)}
            out.append([E[idx[h]] for h in heads])
        return out

    # ---------------- the pre-registered contrast ---------------------------------
    print("[4/6] computing per-head quantities for set S and set N ...", flush=True)
    results = {}
    for name, paths in (("S_seeds_plainCE", SET_S), ("N_random", SET_N)):
        reps = per_head_effects(paths)
        n_rep = len(reps)
        fk_parts, fk_mean, r_norm, cos_rep = [], [], [], []
        for j, h in enumerate(heads):
            wk = WK[h]
            es = [reps[i][j] for i in range(n_rep)]
            fks = [f_know(e, wk) for e in es]
            M = torch.stack(es).mean(dim=0)
            fk_parts.append(f_know(M, wk))
            fk_mean.append(float(np.mean(fks)))
            mean_norm = float(np.mean([float(torch.linalg.norm(e)) for e in es]))
            r_norm.append(float(torch.linalg.norm(M)) / max(mean_norm, 1e-12))
            cs = []
            for a, b in combinations(range(n_rep), 2):
                ua, ub = unit(es[a]), unit(es[b])
                if ua is not None and ub is not None:
                    cs.append(float(ua @ ub))
            cos_rep.append(float(np.median(cs)) if cs else float("nan"))
        results[name] = {
            "f_know_avg": np.array(fk_parts),      # f_know(M) per head
            "f_know_parts": np.array(fk_mean),     # mean_i f_know(e_i) per head
            "r_norm": np.array(r_norm),
            "cos_rep": np.array(cos_rep),
        }
        print(f"      {name}: {n_rep} replicas x {len(heads)} heads", flush=True)

    dS = results["S_seeds_plainCE"]["f_know_avg"] - results["S_seeds_plainCE"]["f_know_parts"]
    dN = results["N_random"]["f_know_avg"] - results["N_random"]["f_know_parts"]
    dd = dS - dN                                     # paired: same heads in both sets

    print(f"[5/6] bootstrap ({args.n_boot} reps over n={len(heads)} heads) ...", flush=True)
    ci_dS = boot_ci(dS, args.n_boot, np.random.default_rng(args.seed))
    ci_dN = boot_ci(dN, args.n_boot, np.random.default_rng(args.seed + 1))
    ci_dd = boot_ci(dd, args.n_boot, np.random.default_rng(args.seed + 2))
    med_fk_avg_S = float(np.median(results["S_seeds_plainCE"]["f_know_avg"]))
    ci_med_S = boot_ci(results["S_seeds_plainCE"]["f_know_avg"], args.n_boot,
                       np.random.default_rng(args.seed + 3), stat=np.median)

    kr_a_pass = not (ci_dd[0] <= 0.0 <= ci_dd[1])
    kr_b_pass = med_fk_avg_S >= floor

    try:
        from scipy.stats import wilcoxon
        w_p = float(wilcoxon(dS, dN).pvalue)
    except Exception:  # noqa: BLE001
        w_p = None

    # ---------------- references + self-test --------------------------------------
    print("[6/6] reference offsets and set D (exploratory) ...", flush=True)
    ref_out = {}
    for name, path in REFS.items():
        pr, E = load_effect(path)
        idx = {q: j for j, q in enumerate(pr)}
        vals = [f_know(E[idx[h]], WK[h]) for h in heads if h in idx]
        ref_out[name] = {"n": len(vals), "median": float(np.median(vals)),
                         "mean": float(np.mean(vals))}
        print(f"      {name:16s} median f_know = {ref_out[name]['median']:.3e}", flush=True)

    d_out = {"supports": {}, "pairwise_shared_heads": {}}
    d_loaded = {}
    for name, path in SET_D.items():
        pr, E = load_effect(path)
        d_loaded[name] = (pr, E)
        d_out["supports"][name] = {"K": len(pr), "n_in_ref48": sum(1 for p in pr if p in WK)}
    for a, b in combinations(SET_D, 2):
        pa, Ea = d_loaded[a]
        pb, Eb = d_loaded[b]
        shared = [h for h in pa if h in set(pb)]
        ia = {q: j for j, q in enumerate(pa)}
        ib = {q: j for j, q in enumerate(pb)}
        cs = []
        for h in shared:
            ua, ub = unit(Ea[ia[h]]), unit(Eb[ib[h]])
            if ua is not None and ub is not None:
                cs.append(float(ua @ ub))
        d_out["pairwise_shared_heads"][f"{a}|{b}"] = {
            "n_shared": len(shared),
            "median_cos": float(np.median(cs)) if cs else None,
        }

    def summ(a):
        return {"median": float(np.median(a)), "mean": float(np.mean(a)),
                "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}

    out = {
        "prereg": "docs/superpowers/plans/2026-08-01-e0-averaging-as-offaxis-filter-prereg.md",
        "config": {"func": args.func, "ridge_frac": args.ridge_frac,
                   "n_boot": args.n_boot, "seed": args.seed,
                   "n_heads": len(heads), "head_dim": head_dim,
                   "set_S": SET_S, "set_N": SET_N},
        "floors": {"random_direction_1_over_d": floor,
                   "cos_rms_random": 1.0 / np.sqrt(head_dim)},
        "build_diagnostics": {k: v for k, v in diag.items()
                              if isinstance(v, (int, float, str, bool))},
        "set_S": {k: summ(v) for k, v in results["S_seeds_plainCE"].items()},
        "set_N": {k: summ(v) for k, v in results["N_random"].items()},
        "kill_rules": {
            "KR_A_power": {
                "delta_S_mean": float(np.mean(dS)), "delta_S_ci95": ci_dS,
                "delta_N_mean": float(np.mean(dN)), "delta_N_ci95": ci_dN,
                "diff_in_diff_mean": float(np.mean(dd)), "diff_in_diff_ci95": ci_dd,
                "wilcoxon_p": w_p,
                "PASS": bool(kr_a_pass),
            },
            "KR_B_materiality": {
                "median_f_know_avg_S": med_fk_avg_S, "ci95": ci_med_S,
                "floor_1_over_d": floor,
                "PASS": bool(kr_b_pass),
            },
        },
        "references": ref_out,
        "set_D_exploratory": d_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------------- verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("E0 VERDICT (kill-rules fixed before the run)")
    print("=" * 78)
    st = ref_out.get("wknow_SELFTEST", {}).get("median")
    print(f"SELF-TEST  f_know(mc_wknow_offset) = {st:.6f}   (must be ~1.0)")
    print(f"           -> pipeline {'OK' if st is not None and st > 0.95 else 'SUSPECT'}")
    print(f"\nfloor (random direction, 1/d) = {floor:.6f}")
    print(f"\nSet S (3 seeds, plain CE):")
    print(f"  cos_rep      median {np.median(results['S_seeds_plainCE']['cos_rep']):+.4f}"
          f"   (random RMS {1/np.sqrt(head_dim):.4f})")
    print(f"  r_norm       median {np.median(results['S_seeds_plainCE']['r_norm']):.4f}"
          f"   (identical=1.000, independent=0.577)")
    print(f"  f_know parts median {np.median(results['S_seeds_plainCE']['f_know_parts']):.3e}")
    print(f"  f_know avg   median {med_fk_avg_S:.3e}")
    print(f"\nKR-A (power)       diff-in-diff {np.mean(dd):+.3e} "
          f"CI95 [{ci_dd[0]:+.3e}, {ci_dd[1]:+.3e}]  -> "
          f"{'PASS' if kr_a_pass else 'FAIL (CI includes 0)'}")
    print(f"KR-B (materiality) median f_know(M) {med_fk_avg_S:.3e} vs floor {floor:.3e}"
          f"  -> {'PASS' if kr_b_pass else 'FAIL (below random floor)'}")
    verdict = ("PROMOTE to E1 (pod)" if (kr_a_pass and kr_b_pass)
               else "LINE CLOSED -- " + ("effect real but immaterial" if kr_a_pass
                                         else "averaging buys nothing over noise"))
    print(f"\n>>> {verdict}")
    print("=" * 78)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
