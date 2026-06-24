"""E3-clean — structural-subspace energy of the off-axis offset's logit-effective write.

The first-pass E3 (`analyze_offaxis_geometry.py`) measured a SINGLE cosine,
cos(offset write, s_struct), where s_struct is the normalized MEAN unembed row over
structural tokens. That mean-direction proxy was INCONCLUSIVE: structural tokens span a
MULTI-dimensional subspace, so collapsing them to one mean direction throws away most of
the structure. This clean version measures the ENERGY FRACTION of the offset's write that
lives inside the structural-token SUBSPACE, against a proper random-subspace null.

Math (logit-channel framing):
  The logit of token t from a RAW residual write r is  logit_t = (g ⊙ embed[t]) · r ,
  with g = (1 + norm) the final-RMSNorm logit gain. Hence the directions whose dot-product
  with r controls the structural tokens' logits are  { g ⊙ embed[t] : t in struct-ids }.
  The structural subspace is their span. Build M = embed[ids] * g, orthonormalize via SVD,
  keep the right-singular vectors above tolerance → basis Q [d_model, r]. The energy of a
  write v inside that subspace is  E_struct(v) = ||Q^T v||^2 / ||v||^2  (a fraction in [0,1]).

We report E_struct for: every per-head write r_i, the aggregate R = sum_i r_i, and (for
contrast) the per-head v_inf-write and v_ret-write. A K=200 random-orthonormal-subspace null
of the SAME rank r gives null_mean (≈ r/d_model), null_std, and z for E_struct(R). The
non-structural remainder r_perp = R - Q(Q^T R) is checked for residual v_inf / v_ret alignment
(harness-viability readout: if structural energy ablates but correctness energy survives, an
s_struct-ablation could be separable).

  python scripts/analyze_offaxis_geometry_subspace.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = "google/gemma-4-31b-it"
MC = "runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt"
FUNC = "runs/functional_directions.pt"
TENS = "runs/logit_lens/g31_tensors.pt"   # embed + o_proj[layers] + norm (from logit-lens fetch)
OUT = "runs/logit_lens/offaxis_geometry_subspace.json"

K_NULL = 200            # random subspaces for the null baseline
ENERGY_COVER = 0.999    # keep right-singular vectors up to this fraction of sum(S^2)
SEED = 0


def unit(x):
    n = torch.linalg.norm(x)
    return x / n if n > 1e-12 else x


def build_s_struct_ids(tok):
    """Structural-token id set — identical to base script's build_s_struct.

    A-Z and '▁'+letter; punctuation and '▁'+punct; Gemma EOS/turn ids {1,50,105,106}.
    """
    ids = set()
    import string
    cands = []
    for c in string.ascii_uppercase:
        cands += [c, "▁" + c]                       # 'A' and '▁A'
    for p in [".", ",", "-", "(", ")", "[", "]", ":", ";", "!", "?", "'", '"', "{", "}", "/"]:
        cands += [p, "▁" + p]
    for t in cands:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    ids |= {1, 50, 105, 106}                              # Gemma EOS/turn stops
    return sorted(ids)


def build_struct_basis(embed, gain, ids, energy_cover=ENERGY_COVER):
    """Orthonormal basis Q [d_model, r] of the structural subspace in RAW residual space.

    The logit-relevant direction for token t is (g ⊙ embed[t]); the subspace is their span.
    M = embed[ids] * g  [n_ids, d_model]; SVD M = U diag(S) Vh; the right-singular vectors
    (rows of Vh) span the row-space of M = exactly that subspace. Keep components up to
    `energy_cover` of sum(S^2) (and strictly > tol). Returns Q (cols = kept Vh rows), rank r,
    kept energy fraction, and the singular values.
    """
    M = (embed[ids].to(torch.float32) * gain)            # [n_ids, d_model]
    # economy SVD: Vh is [min(n,d), d_model]; rows are the orthonormal right-sing vectors.
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    s2 = (S ** 2)
    total = float(s2.sum())
    tol = float(S.max()) * max(M.shape) * 1.1920929e-07   # numpy-style rank tol (eps_f32)
    keep_tol = S > tol
    # energy-coverage cutoff: keep singular values through (and including) the one that first
    # pushes cumulative S^2-energy past `energy_cover`; intersect with the numerical-rank mask.
    cum = torch.cumsum(s2, 0) / total
    first_over = int(torch.argmax((cum > energy_cover).to(torch.int))) if bool((cum > energy_cover).any()) else len(S) - 1
    keep = keep_tol & (torch.arange(len(S)) <= first_over)
    if int(keep.sum()) == 0:
        keep[0] = True
    r = int(keep.sum())
    kept_energy = float(s2[keep].sum() / total)
    Q = Vh[keep].t().contiguous()                         # [d_model, r], orthonormal columns
    return Q, r, kept_energy, S


def e_struct(Q, v):
    """Energy fraction of vector v inside subspace spanned by Q's columns (orthonormal)."""
    vv = float(v @ v)
    if vv <= 1e-30:
        return 0.0
    proj = Q.t() @ v                                       # [r]
    return float((proj @ proj) / vv)


def main():
    print("[geo-sub] loading offsets / functional directions / logit-lens tensors ...", flush=True)
    mc = torch.load(MC, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in p) for p in mc["layer_head_pairs"]]
    alpha = mc["alpha"].to(torch.float32); theta = mc["theta"].to(torch.float32)
    hd = theta.shape[1]

    fd = torch.load(FUNC, weights_only=False, map_location="cpu")
    v_inf = fd["v_inf"].to(torch.float32); v_ret = fd["v_ret"].to(torch.float32)
    captured = [int(x) for x in fd["captured_indices"]]
    cap = {li: ai for ai, li in enumerate(captured)}

    T = torch.load(TENS, weights_only=False, map_location="cpu")
    embed = T["embed"]                                    # [vocab, d_model]
    d_model = embed.shape[1]
    gain = (1.0 + T["norm"].to(torch.float32))            # [d_model]
    oproj = {L: T[f"o_proj_{L}"].to(torch.float32) for L in sorted(set(L for L, _ in pairs)) if f"o_proj_{L}" in T}

    tok = AutoTokenizer.from_pretrained(REPO)
    ids = build_s_struct_ids(tok)
    print(f"[geo-sub] {len(ids)} structural tokens; d_model={d_model}", flush=True)

    print("[geo-sub] building structural subspace basis (SVD of gain*embed[ids]) ...", flush=True)
    Q, r, kept_energy, S = build_struct_basis(embed, gain, ids)
    print(f"[geo-sub] subspace effective rank r={r} (cover {kept_energy:.4f} of S^2); "
          f"null E_struct expectation = r/d_model = {r / d_model:.4f}", flush=True)

    # --- per-head writes and energy fractions ---
    R = torch.zeros(d_model, dtype=torch.float32)          # aggregate offset write
    e_head, e_vinf, e_vret = [], [], []
    Vinf_res = torch.zeros(d_model, dtype=torch.float32)   # aggregate v_inf-write (for r_perp readout)
    Vret_res = torch.zeros(d_model, dtype=torch.float32)   # aggregate v_ret-write
    n_used = 0
    for i, (L, h) in enumerate(pairs):
        if L not in cap or L not in oproj:
            continue
        ai = cap[L]
        off = alpha[i] * theta[i]                          # [hd]
        W = oproj[L][:, h * hd:(h + 1) * hd]               # [d_model, hd]
        r_i = W @ off                                      # residual write [d_model]
        R += r_i
        e_head.append(e_struct(Q, r_i))
        # v_inf / v_ret per-head writes (unit head-dim directions mapped through o_proj)
        vinf_w = W @ unit(v_inf[ai, h]); vret_w = W @ unit(v_ret[ai, h])
        Vinf_res += vinf_w; Vret_res += vret_w
        e_vinf.append(e_struct(Q, vinf_w))
        e_vret.append(e_struct(Q, vret_w))
        n_used += 1
        if n_used % 20 == 0:
            print(f"[geo-sub] [{n_used}] heads processed (last L{L} H{h})", flush=True)

    def ms(x):
        a = np.array(x, dtype=np.float64)
        return dict(mean=float(a.mean()), med=float(np.median(a)),
                    min=float(a.min()), max=float(a.max()), n=int(a.size))

    E_R = e_struct(Q, R)

    # --- NULL baseline: K random orthonormal subspaces of rank r in d_model ---
    print(f"[geo-sub] drawing {K_NULL} random rank-{r} subspaces for the null ...", flush=True)
    rng = np.random.default_rng(SEED)
    R_np = R.numpy().astype(np.float64)
    R_norm2 = float(R_np @ R_np)
    null_vals = np.empty(K_NULL, dtype=np.float64)
    for k in range(K_NULL):
        G = rng.standard_normal((d_model, r))
        Qn, _ = np.linalg.qr(G)                            # [d_model, r] orthonormal cols
        proj = Qn.T @ R_np                                 # [r]
        null_vals[k] = float(proj @ proj) / R_norm2 if R_norm2 > 0 else 0.0
        if (k + 1) % 50 == 0:
            print(f"[geo-sub] null [{k + 1}/{K_NULL}]", flush=True)
    null_mean = float(null_vals.mean()); null_std = float(null_vals.std(ddof=1))
    z_R = (E_R - null_mean) / null_std if null_std > 1e-12 else float("inf")

    # --- harness-viability readout: non-structural remainder of R ---
    proj_R = Q @ (Q.t() @ R)                               # component of R inside subspace
    r_perp = R - proj_R                                    # non-structural remainder
    frac_nonstruct = 1.0 - E_R                             # = ||r_perp||^2 / ||R||^2
    # does the remainder still align with correctness/retrieval directions?
    # Both sides live in RAW d_model space: the subspace Q and the ablation are
    # raw writes, and Vinf_res/Vret_res are raw residual writes, so this is a
    # raw-space alignment (no gain mixing — cf. base script gain-scales BOTH sides).
    cos_perp_vinf = float(unit(r_perp) @ unit(Vinf_res))
    cos_perp_vret = float(unit(r_perp) @ unit(Vret_res))
    # for reference: same alignment on the FULL R (pre-ablation)
    cos_R_vinf = float(unit(R) @ unit(Vinf_res))
    cos_R_vret = float(unit(R) @ unit(Vret_res))

    res = {
        "config": {
            "n_struct_tokens": len(ids), "d_model": d_model,
            "subspace_rank_r": r, "kept_energy_cover": kept_energy,
            "energy_cover_target": ENERGY_COVER, "k_null": K_NULL, "seed": SEED,
            "n_heads_used": n_used, "head_dim": hd,
        },
        "E3_struct_subspace_energy": {
            "E_struct_aggregate_R": E_R,
            "null_mean": null_mean, "null_std": null_std,
            "null_expectation_r_over_d": r / d_model,
            "z_R": z_R,
            "per_head_E_struct_write": ms(e_head),
            "per_head_E_struct_v_inf_write": ms(e_vinf),
            "per_head_E_struct_v_ret_write": ms(e_vret),
        },
        "harness_viability_readout": {
            "frac_nonstructural_R": frac_nonstruct,
            "cos_rperp_v_inf": cos_perp_vinf,
            "cos_rperp_v_ret": cos_perp_vret,
            "cos_R_v_inf_reference": cos_R_vinf,
            "cos_R_v_ret_reference": cos_R_vret,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print("\n=== E3-clean — structural-subspace energy of the offset write ===")
    print(f"  subspace rank r                       = {r}  (cover {kept_energy:.4f}, null E[r/d]={r / d_model:.4f})")
    print(f"  E_struct(aggregate R)                 = {E_R:.4f}")
    print(f"  null mean / std                       = {null_mean:.4f} / {null_std:.4f}")
    print(f"  z (R vs random subspace null)         = {z_R:+.2f}")
    print(f"  per-head E_struct(offset write)       : {res['E3_struct_subspace_energy']['per_head_E_struct_write']}")
    print(f"  per-head E_struct(v_inf write)        : {res['E3_struct_subspace_energy']['per_head_E_struct_v_inf_write']}")
    print(f"  per-head E_struct(v_ret write)        : {res['E3_struct_subspace_energy']['per_head_E_struct_v_ret_write']}")
    print("\n=== harness-viability readout (non-structural remainder of R) ===")
    print(f"  frac non-structural (1 - E_struct(R)) = {frac_nonstruct:.4f}")
    print(f"  cos(r_perp, gain*v_inf-write-agg)     = {cos_perp_vinf:+.4f}   (full R: {cos_R_vinf:+.4f})")
    print(f"  cos(r_perp, gain*v_ret-write-agg)     = {cos_perp_vret:+.4f}   (full R: {cos_R_vret:+.4f})")
    print("\nREAD: E_struct(R) >> null with high z  =>  the offset's READABLE logit effect lives "
          "inside the structural-token subspace (uppercase/punct/EOS) — i.e. structural-token "
          "suppression, the EOS/format-collapse face. That supports mechanism C (format-machinery "
          "suppression) / F (last-token de-collapse), and is the SUBSPACE-clean upgrade of the "
          "inconclusive single-cosine E3.\n"
          "      If, after removing the structural component, the remainder r_perp STILL aligns "
          "with v_inf/v_ret (cos materially >0), then correctness energy survives an s_struct "
          "ablation => the two are separable => an s_struct-ablation harness is geometrically "
          "viable (kill the format-suppression face, keep the correctness push). If r_perp loses "
          "the v_inf/v_ret alignment, the structural face and the correctness push are entangled "
          "in the SAME subspace => not separable by ablation alone.")
    print(f"[geo-sub] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
