"""Build the W_know (Fisher-LDA / whitened on-axis) offset variant.

Pre-reg: docs/superpowers/specs/2026-06-23-wknow-onaxis-control-design.md.

WHY THIS EXISTS
---------------
The causal-of-offset test (B, make_vinf_offset.py) replaced the trained mc
direction with the *raw mass-mean* inference-correctness direction v_inf
(= mu_correct - mu_wrong, unit) and found it recovers only ~11% of the ARC lift
=> "OOD hack". Exp-A then raised reviewer-attack-1: mass-mean is a WEAK estimate
of the on-axis direction; a stronger (whitened / trained) correctness direction
might recover more. W_know answers it with the strongest on-axis reference
buildable locally (€0): the Fisher-LDA direction.

THE DIRECTION
-------------
Per adapter head, W_know = Sigma_w^{-1} (mu_correct - mu_wrong), unit-normalized,
where Sigma_w is the pooled within-class covariance stored in
functional_directions.pt (`cov_inf`, eligible heads only). NOTE: cov_inf is
centered on the POOLED mean (park_whiten style) = within-class scatter + a rank-1
between-class term c*v v^T. By Sherman-Morrison, cov_inf^{-1} v is PARALLEL to
Sigma_w^{-1} v (differ by a scalar), so after unit-normalization the W_know
direction equals the proper Fisher-LDA direction. This is exactly the Mahalanobis
("whitened") space in which theta_mc was found ⊥ v_inf (cos_whitened -0.003).

    offset_wknow[L,h] = || alpha_mc[i] * theta_mc[i] ||  *  W_know_unit[L,h]   (sign +)

Same applied norm as mc (isolates DIRECTION, not magnitude), byte-compatible with
install_lofit_hooks so run_lm_eval_v7 --offsets loads it unchanged.

READ
----
If WKNOW lift << MC lift (like v_inf) -> the +35 is off-axis even vs the strongest
local on-axis reference -> off-axis ROBUST (reviewer-attack-1 defused).
If WKNOW lift >> v_inf lift (approaches MC) -> mass-mean hid alignment; part of the
ARC lift is real (whitened) correctness -> ARC verdict flips toward real.

METHODOLOGICAL GUARD (circularity): W_know is built from BASE activations (no
adapter at capture) predicting GROUND-TRUTH correctness labels — never adapter
logits/behavior. It cannot "re-find" the adapter's direction; it can only express
the optimal on-base correctness direction. The test is therefore clean.

CPU/local (€0). Run it, verify the diagnostics, commit the tiny blob, then eval.
"""
import json
import os

import numpy as np
import torch

ROOT = os.environ.get("MSAP_ROOT", "/workspace/MSAP")
MC = os.path.join(ROOT, "runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt")
# E2 output. Local default mirrors where it actually lives; pod/HF override via env.
FUNC = os.environ.get("FUNC_DIRS", os.path.join(ROOT, "runs/functional_directions.pt"))
OUT_DIR = os.path.join(ROOT, "runs/vinf_causal")
OUT = os.path.join(OUT_DIR, "mc_wknow_offset.pt")
ELIG_ACC = float(os.environ.get("VINF_ELIG_ACC", "0.60"))
# Shrinkage as a fraction of the mean diagonal of cov (ridge). cond(cov) max ~8e3
# => 1e-2 is safely conditioned; the preview showed cos(LDA,theta_mc) is stable
# across 1e-3..1e-1 so the verdict does not hinge on this knob.
SHRINK = float(os.environ.get("WKNOW_SHRINKAGE", "1e-2"))
os.makedirs(OUT_DIR, exist_ok=True)


def _resolve_func():
    if os.path.exists(FUNC):
        return FUNC
    for c in [os.path.join(ROOT, "runs/oq1_functional_axis/functional_directions.pt"),
              os.path.join(ROOT, "runs/functional_directions.pt")]:
        if os.path.exists(c):
            return c
    raise SystemExit(
        f"FATAL: functional_directions.pt not found ({FUNC}). Set FUNC_DIRS=<path> "
        f"(E2 output; pod: pull HF sebacascardo87/msap-oq1-vg-20260622).")


def main():
    func = _resolve_func()
    mc = torch.load(MC, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in p) for p in mc["layer_head_pairs"]]
    alpha_mc = mc["alpha"].clone().to(torch.float32)
    theta_mc_raw = mc["theta"].clone().to(torch.float32)
    n = len(pairs)
    assert n == alpha_mc.shape[0] == theta_mc_raw.shape[0], "mc blob shape mismatch"

    fd = torch.load(func, weights_only=False, map_location="cpu")
    mu_c = fd["mu_inf_correct"].to(torch.float64)        # [n_lay, nH, D]
    mu_w = fd["mu_inf_wrong"].to(torch.float64)
    v_inf = fd["v_inf"].to(torch.float64)                # mass-mean fallback
    theta_grid = fd["theta_mc"].to(torch.float64)        # applied alpha*theta on grid
    cov_inf = fd["cov_inf"]                               # dict (layer,head)->[D,D]
    acc_inf = fd.get("acc_inf")
    captured = [int(x) for x in fd["captured_indices"]]
    cap_index = {li: ai for ai, li in enumerate(captured)}
    head_dim = int(fd.get("head_dim", v_inf.shape[-1]))

    new_alpha = alpha_mc.clone()
    new_theta = theta_mc_raw.clone()
    n_lda = n_massmean_fb = n_skipped = n_null = n_lowacc = 0
    cos_wknow_theta, cos_wknow_vinf, norm_errs, conds, acc_list = [], [], [], [], []

    for i, (li, hi) in enumerate(pairs):
        if li not in cap_index:
            n_skipped += 1            # layer not in E2 grid — keep mc, flag
            continue
        ai = cap_index[li]
        v = (mu_c[ai, hi] - mu_w[ai, hi])               # mass-mean numerator
        off_mc = alpha_mc[i].to(torch.float64) * theta_mc_raw[i].to(torch.float64)
        scale = float(torch.linalg.norm(off_mc))        # norm to match

        cov = cov_inf.get((li, hi))
        if cov is not None:
            cov = cov.to(torch.float64)
            ridge = SHRINK * torch.diag(cov).mean()
            cov_r = cov + ridge * torch.eye(head_dim, dtype=torch.float64)
            w = torch.linalg.solve(cov_r, v)            # Fisher-LDA direction
            ev = torch.linalg.eigvalsh(cov)
            conds.append(float(ev[-1] / max(ev[0].item(), 1e-12)))
            src = "lda"
        else:
            w = v_inf[ai, hi].clone()                   # fallback: mass-mean
            src = "massmean_fb"

        wn = float(torch.linalg.norm(w))
        if wn < 1e-8:
            new_alpha[i] = 0.0
            new_theta[i] = torch.zeros(head_dim, dtype=torch.float32)
            n_null += 1
            continue
        w_unit = w / wn
        new_alpha[i] = scale
        new_theta[i] = w_unit.to(torch.float32)
        if src == "lda":
            n_lda += 1
        else:
            n_massmean_fb += 1

        # diagnostics
        off_new = new_alpha[i].to(torch.float64) * new_theta[i].to(torch.float64)
        norm_errs.append(abs(float(torch.linalg.norm(off_new)) - scale))
        th = theta_grid[ai, hi]
        if float(torch.linalg.norm(th)) > 1e-9:
            cos_wknow_theta.append(float((w_unit @ (th / torch.linalg.norm(th)))))
        vmm = v_inf[ai, hi]
        if float(torch.linalg.norm(vmm)) > 1e-9:
            cos_wknow_vinf.append(float((w_unit @ (vmm / torch.linalg.norm(vmm)))))
        if acc_inf is not None:
            a = float(acc_inf[ai, hi]); acc_list.append(a)
            if a < ELIG_ACC:
                n_lowacc += 1

    out = {**mc, "alpha": new_alpha, "theta": new_theta,
           "config": {**mc.get("config", {}), "wknow_variant": "mc_wknow_lda_onaxis",
                      "built_from": {"mc": MC, "func": func}, "sign": "+w_know",
                      "shrinkage_frac": SHRINK, "direction": "fisher_lda_cov_inf_inv_massmean",
                      "norm_matched_to": "applied_mc_offset"}}
    torch.save(out, OUT)
    chk = torch.load(OUT, weights_only=False, map_location="cpu")
    assert len(chk["layer_head_pairs"]) == n, "pairs count changed!"

    def ms(x):
        a = np.array(x)
        return f"mean {a.mean():+.4f} med {np.median(a):+.4f} min {a.min():+.4f} max {a.max():+.4f}"

    print("=" * 68)
    print("make_wknow_offset — Fisher-LDA on-axis (W_know) offset, norm-matched to mc")
    print("=" * 68)
    print(f"  mc offsets : {MC}  ({n} heads)")
    print(f"  func dirs  : {func}")
    print(f"  shrinkage  : {SHRINK} (ridge frac of mean diag)")
    print(f"  LDA heads  : {n_lda}/{n}   massmean-fallback: {n_massmean_fb}   "
          f"skipped(kept mc): {n_skipped}   zeroed(null): {n_null}")
    if acc_list:
        print(f"  acc_inf    : mean {np.mean(acc_list):.3f}  {n_lowacc}/{len(acc_list)} < {ELIG_ACC}")
    if conds:
        print(f"  cov cond   : mean {np.mean(conds):.1f}  max {max(conds):.1f}")
    if norm_errs:
        print(f"  norm-match max abs err : {max(norm_errs):.2e}  (should be ~0)")
    if cos_wknow_vinf:
        print(f"  cos(W_know, v_inf)     : {ms(cos_wknow_vinf)}  (rotation by whitening; <1 => genuinely different ref)")
    if cos_wknow_theta:
        print(f"  cos(W_know, theta_mc)  : {ms(cos_wknow_theta)}  <-- PREVIEW: ~0 => off-axis robust expected")
    print(f"\n  wrote {OUT}")
    print(json.dumps({"out": OUT, "n_heads": n, "n_lda": n_lda,
                      "n_massmean_fb": n_massmean_fb, "n_skipped_kept_mc": n_skipped,
                      "n_zeroed": n_null, "shrinkage": SHRINK,
                      "cos_wknow_theta_mean": float(np.mean(cos_wknow_theta)) if cos_wknow_theta else None,
                      "cos_wknow_vinf_mean": float(np.mean(cos_wknow_vinf)) if cos_wknow_vinf else None}))


if __name__ == "__main__":
    main()
