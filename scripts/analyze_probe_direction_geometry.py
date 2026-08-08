"""M1 — READ-axis vs WRITE-axis: cos(probe_direction, theta_mc) and cos(probe_direction, v_inf).

Closes the read/=write synthesis GEOMETRICALLY (2026-06-29): the probe READS correctness
(in-space, AUC 0.955 / selector-acc 0.946); the adapter WRITES the lift along theta_mc (off-axis,
cos(theta_mc,v_inf)=-0.003 whitened). If the probe's decision direction is ~aligned with v_inf
(the correctness axis) and ~orthogonal to theta_mc (what the adapter writes), then "read on-axis,
write off-axis" is ONE measured number, not two separate experiments (probe vs W_know).

SPACE IS LOAD-BEARING (workflow-verified, both reported):
  RAW  o_proj-input per-head (head_dim=256, identity metric) — where probe/theta_mc/v_inf live crude.
  WHITENED Mahalanobis per-head, M = inv(cov_inf + ridge), ridge=1e-2*mean(diag(cov)) (same as make_wknow).
  The canonical -0.003 is WHITENED -> only the whitened cosine compares to it.
GATE (recalibrated): the internal check cos_whit(theta_mc, v_inf) must land IN the random-null band
  (|cos| << 1/sqrt(256)=0.0625), NOT reproduce -0.003 to three decimals (sign/decimal is convention
  noise: local check gave +0.0010 per-head / +0.00385 pooled — quasi-null, pipeline SANE).
PC-BASELINE (load-bearing control): if v_inf lives trivially in the acts' top PCs, a high cos(probe,v_inf)
  is by construction (the probe passes through PCA). Report cos(PC1, v_inf) and cos(probe, PC1): the
  result is informative only if cos(probe,v_inf) >> cos(PC1,v_inf).
Decisive aggregation = PER-HEAD distribution (mean/median + null band); pooled global is SECONDARY.
SIGNED cos for v_inf (read correctness in the +mu_correct-mu_wrong sense), |cos| for the null-band contrast.

Predicted: cos(probe,v_inf) HIGH (|cos|~0.2-0.5 raw, signed +, >> null band), cos(probe,theta_mc) LOW
(within/near null band). Falsifier: cos(probe,theta_mc) high (>0.15 whitened) -> read=write, chain falls;
OR cos(probe,v_inf) in null band -> the 0.955 reader is a 3rd axis, "read on-axis" false.

CPU-local, EUR0. Usage:
  python scripts/analyze_probe_direction_geometry.py \
      --acts runs/vinf_causal/coldmc_perhead_arc_challenge.pt --C 0.03
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    import torch
except Exception:
    torch = None
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
MC = ROOT / "runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt"
FUNC_CANDS = [ROOT / "runs/functional_directions.pt",
              ROOT / "runs/oq1_functional_axis/functional_directions.pt"]
RIDGE_FRAC = 1e-2


def fit_probe_dir(X, y, C, pca_k, seed):
    """Fit ONE probe on ALL data; return its decision direction in RAW-12288 space,
    plus the scaler/pca (for the PC-baseline). w_raw = (pca.components_.T @ coef)/scale."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    pca = PCA(n_components=pca_k, random_state=seed).fit(Xs)
    Xp = pca.transform(Xs)
    clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=2000).fit(Xp, y)
    w = pca.components_.T @ clf.coef_.ravel()        # raw-12288 in scaled space
    w_raw = w / sc.scale_                              # de-scale -> dz/dX_raw
    return w_raw, sc, pca, clf


def whiten_M(cov, head_dim):
    ridge = RIDGE_FRAC * float(np.mean(np.diag(cov)))
    return np.linalg.inv(cov + ridge * np.eye(head_dim))


def cos_raw(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def cos_whit(a, b, M):
    qa, qb, qab = a @ M @ a, b @ M @ b, a @ M @ b
    return float(qab / np.sqrt(qa * qb)) if qa > 1e-18 and qb > 1e-18 else 0.0


def dist(x):
    a = np.array(x, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()),
                absmean=float(np.abs(a).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--C", type=float, default=0.03, help="winning C from the held-out run")
    ap.add_argument("--pca", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--fold-stability", action="store_true", default=True)
    ap.add_argument("--no-fold-stability", dest="fold_stability", action="store_false")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if torch is None:
        raise SystemExit("torch required")

    st = torch.load(args.acts, weights_only=False, map_location="cpu")
    A = st["acts_adapter"]                                   # (n_prompts, 48, 256)
    nP, nC, hd = A.shape
    X = A.reshape(nP, nC * hd).numpy().astype(np.float64)
    y = np.array([m["is_gold"] for m in st["prompt_meta"]], dtype=np.int64)
    groups = np.array([m["item_idx"] for m in st["prompt_meta"]], dtype=np.int64)
    cells = [tuple(int(z) for z in c) for c in st["adapter_cells_layer_head"]]

    mc = torch.load(MC, weights_only=False, map_location="cpu")
    mc_pairs = [tuple(int(z) for z in p) for p in mc["layer_head_pairs"]]
    assert mc_pairs == cells, "cell order mismatch acts vs offsets_mc"
    theta = mc["theta"].to(torch.float64).numpy()           # (48,256) NON-unit (norm ~3.79)
    theta_unit = theta / np.linalg.norm(theta, axis=1, keepdims=True)

    func = next((p for p in FUNC_CANDS if p.exists()), None)
    if func is None:
        raise SystemExit("functional_directions.pt not found")
    fd = torch.load(func, weights_only=False, map_location="cpu")
    v_inf = fd["v_inf"].to(torch.float64).numpy()
    captured = [int(z) for z in fd["captured_indices"]]
    cap = {L: ai for ai, L in enumerate(captured)}
    cov_inf = fd["cov_inf"]

    print("=" * 80, flush=True)
    print(f"M1 PROBE-DIRECTION GEOMETRY | cells={nC} | rows={nP} | C={args.C} | pca={args.pca}", flush=True)
    print("=" * 80, flush=True)

    # --- fit ONE probe, recover raw decision direction, per-head ---
    w_raw, sc, pca, clf = fit_probe_dir(X, y, args.C, args.pca, args.seed)
    P = w_raw.reshape(nC, hd)                                # (48,256) per-head probe contribution

    v_unit = np.zeros((nC, hd))
    for i, (L, h) in enumerate(cells):
        v = v_inf[cap[L], h]
        v_unit[i] = v / np.linalg.norm(v)

    # --- per-head cosines, raw + whitened ---
    cr_v, cr_t = [], []          # raw cos(probe, v_inf) SIGNED ; cos(probe, theta) |.|
    cw_v, cw_t, cw_tv = [], [], []
    for i, (L, h) in enumerate(cells):
        cr_v.append(cos_raw(P[i], v_unit[i]))               # signed (read-direction)
        cr_t.append(abs(cos_raw(P[i], theta_unit[i])))      # |.| (null-band contrast)
        M = whiten_M(np.asarray(cov_inf[(L, h)], dtype=np.float64), hd)
        cw_v.append(cos_whit(P[i], v_unit[i], M))
        cw_t.append(abs(cos_whit(P[i], theta_unit[i], M)))
        cw_tv.append(cos_whit(theta_unit[i], v_unit[i], M)) # internal check theta⊥v

    # --- pooled global (secondary) ---
    Pp = P.reshape(-1); vp = v_unit.reshape(-1); tp = theta_unit.reshape(-1)
    pooled_raw_v = cos_raw(Pp, vp); pooled_raw_t = abs(cos_raw(Pp, tp))

    # --- random-direction null band (per-head, raw) ---
    rng = np.random.default_rng(args.seed)
    null_abs = []
    for _ in range(args.n_null):
        r = rng.standard_normal((nC, hd))
        c = [abs(cos_raw(r[i], v_unit[i])) for i in range(nC)]
        null_abs.append(np.mean(c))
    null_lo, null_hi = float(np.percentile(null_abs, 2.5)), float(np.percentile(null_abs, 97.5))
    null_band = float(1.0 / np.sqrt(hd))

    # --- PC-baseline: is v_inf trivially in PC1? ---
    pc1_raw = (pca.components_[0] / sc.scale_).reshape(nC, hd)
    pc1_v = np.mean([cos_raw(pc1_raw[i], v_unit[i]) for i in range(nC)])
    probe_pc1 = cos_raw(Pp, pc1_raw.reshape(-1))

    # --- fold-stability ---
    fold_stab = None
    if args.fold_stability:
        from sklearn.model_selection import GroupKFold
        dirs = []
        for tr, _ in GroupKFold(n_splits=5).split(X, y, groups):
            wf, _, _, _ = fit_probe_dir(X[tr], y[tr], args.C, args.pca, args.seed)
            dirs.append(wf / np.linalg.norm(wf))
        mean_dir = np.mean(dirs, axis=0); mean_dir /= np.linalg.norm(mean_dir)
        fold_stab = cos_raw(w_raw / np.linalg.norm(w_raw), mean_dir)

    # --- verdict ---
    check_ok = abs(np.mean(cw_tv)) < null_band              # theta⊥v whitened in null band
    read_on = (np.mean(cr_v) > null_hi) and (np.mean(cr_v) > pc1_v)   # signed, above null AND above PC1
    write_off = (np.mean(cr_t) <= null_hi * 1.5)            # |cos(probe,theta)| ~ null band
    verdict = ("READ≠WRITE GEOMETRIC: probe reads ~v_inf (cos signed >> null AND > PC1), "
               "writes ~⊥theta_mc (|cos| ~ null band) — read on-axis, write off-axis is ONE measure"
               if (read_on and write_off and check_ok) else
               "DOES NOT CLEANLY CLOSE: check the per-head vs pooled, the PC-baseline, and fold-stability")
    if not check_ok:
        verdict = "WHITENING SUSPECT: cos_whit(theta,v_inf) NOT in null band — re-derive before trusting whitened"

    print(f"  internal check cos_whit(theta_mc, v_inf): mean {np.mean(cw_tv):+.4f}  "
          f"(must be < null band {null_band:.4f}) -> {'OK quasi-null' if check_ok else 'SUSPECT'}", flush=True)
    print(f"  null band (random dir): |cos| 1/sqrt(d)={null_band:.4f}, CI[{null_lo:.4f},{null_hi:.4f}]", flush=True)
    print(f"\n  === RAW per-head (DECISIVE) ===", flush=True)
    print(f"      cos(probe, v_inf)  SIGNED : {json.dumps(dist(cr_v))}", flush=True)
    print(f"      cos(probe, theta)  |.|    : {json.dumps(dist(cr_t))}", flush=True)
    print(f"      pooled raw: v_inf {pooled_raw_v:+.4f} | theta |{pooled_raw_t:.4f}|", flush=True)
    print(f"  === WHITENED per-head ===", flush=True)
    print(f"      cos_whit(probe, v_inf) SIGNED : {json.dumps(dist(cw_v))}", flush=True)
    print(f"      cos_whit(probe, theta) |.|    : {json.dumps(dist(cw_t))}", flush=True)
    print(f"  === PC-baseline (is v_inf trivially in PC1?) ===", flush=True)
    print(f"      cos(PC1, v_inf) per-head mean : {pc1_v:+.4f}   cos(probe, PC1): {probe_pc1:+.4f}", flush=True)
    print(f"      informative iff cos(probe,v_inf)={np.mean(cr_v):+.4f} >> cos(PC1,v_inf)={pc1_v:+.4f}", flush=True)
    if fold_stab is not None:
        print(f"  === fold-stability: cos(fit-once, mean-5-fold-dir) = {fold_stab:.4f}  "
              f"{'OK (>0.9)' if fold_stab > 0.9 else 'FRAGILE (<0.9): fit-once may not represent the 0.955 reader'}", flush=True)
    print(f"\n  === VERDICT ===\n      {verdict}", flush=True)

    payload = dict(C=args.C, pca=args.pca, n_cells=nC,
                   null_band=null_band, null_ci=[null_lo, null_hi],
                   check_cos_whit_theta_v=float(np.mean(cw_tv)), check_ok=bool(check_ok),
                   raw_cos_probe_vinf_signed=dist(cr_v), raw_cos_probe_theta_abs=dist(cr_t),
                   whit_cos_probe_vinf_signed=dist(cw_v), whit_cos_probe_theta_abs=dist(cw_t),
                   pooled_raw_vinf=pooled_raw_v, pooled_raw_theta_abs=pooled_raw_t,
                   pc_baseline=dict(cos_pc1_vinf=float(pc1_v), cos_probe_pc1=float(probe_pc1)),
                   fold_stability=fold_stab, verdict=verdict)
    outp = args.out or (args.acts.parent / "probe_direction_geometry.json")
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {outp}", flush=True)


if __name__ == "__main__":
    main()
