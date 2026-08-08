"""Null-space vs row-space decomposition of the V7-mc offset (theta_mc).

WHY (2026-06-30): the +35 cold-MC lift is off-axis to correctness. The open
mechanistic question is WHETHER the offset's DIRECT effect on the vocabulary is a
UNIFORM temperature shift (the Stolfo et al. NeurIPS-2024 "confidence regulation
neuron" / unembedding-null-space mechanism: scales all logits together, changes
entropy/temperature NOT the argmax) or a RANKING change (moves differences between
specific tokens = content/symbol channel, Wong/Lieberum two-stage symbol-binding).

This DECIDES the paper lead:
  - uniform_shift dominant  -> "lossy readout via a temperature/null channel" gets a
    NAMED prior-art mechanism (Stolfo) -> the readout framing is defensible as a lead.
  - ranking dominant        -> the effect is content/symbol re-ranking -> lead with the
    attribution spectrum / off-axis-of-degree (the channel-claim is scoop-adjacent to
    Confidence Manifold 2602.08159).

METHOD (reuses logit_lens_theta_mc.fetch_tensors; NO GPU, NO full model load):
  Per adapter head (L,h): residual write r = o_proj.weight[L][:, h*hd:(h+1)*hd] @ (alpha*theta).
  Direct logit effect (logit-lens approx): delta = W_U @ ((1+norm_w) * r) = M @ r, M = embed*gain.
  Decompose delta in VOCAB space:
    uniform component  = mean(delta) * 1_vocab      (parallel to 1; softmax-invariant shift = temperature/null)
    ranking  component = delta - uniform            (perpendicular to 1; changes the argmax)
    uniform_shift_frac = ||uniform|| / ||delta||,  ranking_frac = ||ranking|| / ||delta||
  Random baseline: norm-matched random residual writes -> expected uniform_shift_frac ~ 1/sqrt(vocab).
  Residual-space temperature direction d* = argmin_r ||M r - 1_vocab|| (the residual dir
  whose logit-image is most uniform); report cos(agg_residual, d*) and how uniform M d* is.

Usage (local CPU):  python scripts/analyze_theta_mc_null_vs_row.py
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logit_lens_theta_mc import fetch_tensors, categorize, MC  # noqa: E402

OUT = "runs/logit_lens/theta_mc_null_vs_row.json"
N_RAND = 32
RIDGE = 1e-2


def decompose_vocab(delta):
    """Split a vocab-space logit delta into uniform (||1) vs ranking (perp 1) parts."""
    vocab = delta.shape[0]
    total = float(delta.norm())
    mean = float(delta.mean())
    uniform_norm = abs(mean) * math.sqrt(vocab)           # ||mean * ones||
    perp = delta - mean
    ranking_norm = float(perp.norm())
    if total < 1e-12:
        return {"uniform_shift_frac": 0.0, "ranking_frac": 0.0, "total": 0.0}
    return {"uniform_shift_frac": uniform_norm / total,
            "ranking_frac": ranking_norm / total,
            "mean_shift": mean, "total": total}


def main():
    mc = torch.load(MC, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in p) for p in mc["layer_head_pairs"]]
    alpha = mc["alpha"].to(torch.float32)
    theta = mc["theta"].to(torch.float32)
    hd = theta.shape[1]
    layers = sorted(set(L for L, _ in pairs))
    print(f"[nvr] {len(pairs)} adapter heads, head_dim={hd}, layers={layers}", flush=True)

    T = fetch_tensors(layers)
    embed = T["embed"].to(torch.float32)                  # [vocab, d_model] = W_U (tied)
    gain = (1.0 + T["norm"].to(torch.float32))            # Gemma RMSNorm (1+w)
    vocab, d_model = embed.shape
    oproj = {L: T[f"o_proj_{L}"].to(torch.float32) for L in layers}
    print(f"[nvr] vocab={vocab} d_model={d_model}; random baseline ~1/sqrt(vocab)="
          f"{1.0/math.sqrt(vocab):.5f}", flush=True)

    # M = embed * gain : the linear map residual r -> direct logit delta
    gvec = gain  # [d_model]

    def logit_of(r):
        return embed @ (gvec * r)

    # --- aggregate net write of all 48 heads + per-head ---
    agg = torch.zeros(d_model, dtype=torch.float32)
    per_head = []
    for i, (L, h) in enumerate(pairs):
        W = oproj[L]
        total_q = W.shape[1]
        nh = total_q // hd
        if h >= nh or total_q % hd != 0:
            print(f"[nvr]   SKIP L{L}h{h}", flush=True)
            continue
        off = alpha[i] * theta[i]
        r = W[:, h * hd:(h + 1) * hd] @ off
        agg += r
        d = decompose_vocab(logit_of(r))
        per_head.append({"layer": L, "head": h, "resid_norm": round(float(r.norm()), 3),
                         "uniform_shift_frac": round(d["uniform_shift_frac"], 4),
                         "ranking_frac": round(d["ranking_frac"], 4)})
        if (i + 1) % 12 == 0:
            print(f"[nvr]   head {i+1}/{len(pairs)}", flush=True)

    agg_delta = logit_of(agg)
    agg_dec = decompose_vocab(agg_delta)
    print(f"[nvr] aggregate: uniform_shift_frac={agg_dec['uniform_shift_frac']:.4f}  "
          f"ranking_frac={agg_dec['ranking_frac']:.4f}  |agg_resid|={float(agg.norm()):.2f}",
          flush=True)

    # --- random baseline (norm-matched residual writes) ---
    g = torch.Generator().manual_seed(0)
    rand_fracs = []
    agg_norm = float(agg.norm())
    for s in range(N_RAND):
        rr = torch.randn(d_model, generator=g, dtype=torch.float32)
        rr = rr / rr.norm() * agg_norm
        rand_fracs.append(decompose_vocab(logit_of(rr))["uniform_shift_frac"])
    rand_fracs = torch.tensor(rand_fracs)
    print(f"[nvr] random uniform_shift_frac: mean={rand_fracs.mean():.5f} "
          f"sd={rand_fracs.std():.5f} max={rand_fracs.max():.5f} (n={N_RAND})", flush=True)

    # --- residual-space temperature direction d* = argmin ||M r - 1_vocab|| ---
    # normal equations with ridge: (M^T M + lam I) x = M^T 1 ; M = embed*gvec
    print("[nvr] computing Gram (embed^T embed) for d* ...", flush=True)
    gram = embed.T @ embed                                # [d_model, d_model]
    AtA = gram * gvec.unsqueeze(0) * gvec.unsqueeze(1)
    Atb = gvec * embed.sum(0)                             # M^T @ ones_vocab
    AtA += RIDGE * torch.eye(d_model)
    dstar = torch.linalg.solve(AtA, Atb)
    dstar = dstar / dstar.norm()
    Md = logit_of(dstar)
    dstar_uniformity = decompose_vocab(Md)["uniform_shift_frac"]   # how uniform d*'s image is
    cos_agg_dstar = float(torch.dot(agg / agg.norm(), dstar))
    print(f"[nvr] d* (residual temperature dir): its image is {dstar_uniformity:.4f} uniform; "
          f"cos(agg_resid, d*)={cos_agg_dstar:.4f}", flush=True)

    # --- verdict ---
    u = agg_dec["uniform_shift_frac"]
    base = float(rand_fracs.mean()) + 3 * float(rand_fracs.std())
    if u > 0.30 and u > 5 * base:
        verdict = ("TEMPERATURE-DOMINANT: the offset's direct logit effect is largely a "
                   "uniform shift (Stolfo null-space/temperature signature) -> 'lossy "
                   "readout via temperature channel' is a NAMED-mechanism lead.")
    elif u < 0.05 or u < 3 * base:
        verdict = ("RANKING-DOMINANT: the offset moves token DIFFERENCES, not a uniform "
                   "shift -> NOT a simple temperature mechanism -> lead with attribution "
                   "spectrum / off-axis-of-degree (content/symbol channel).")
    else:
        verdict = ("MIXED: a non-trivial uniform component above the random baseline but "
                   "ranking still carries most of the effect -> report as a partial "
                   "temperature contribution, do not lead on it alone.")
    print("\n" + "=" * 70 + f"\nVERDICT: {verdict}\n" + "=" * 70, flush=True)

    per_head_sorted = sorted(per_head, key=lambda x: -x["uniform_shift_frac"])
    print("\nPer-head uniform_shift_frac (top 8 most temperature-like):")
    for ph in per_head_sorted[:8]:
        print(f"   L{ph['layer']:2d}h{ph['head']:2d} uniform={ph['uniform_shift_frac']:.4f} "
              f"ranking={ph['ranking_frac']:.4f} |r|={ph['resid_norm']:.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({
        "aggregate": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in agg_dec.items()},
        "agg_resid_norm": round(agg_norm, 4),
        "random_baseline_uniform_frac": {"mean": float(rand_fracs.mean()),
                                         "sd": float(rand_fracs.std()),
                                         "max": float(rand_fracs.max()), "n": N_RAND},
        "analytic_random_expectation": 1.0 / math.sqrt(vocab),
        "dstar_image_uniformity": round(dstar_uniformity, 4),
        "cos_agg_dstar": round(cos_agg_dstar, 4),
        "verdict": verdict,
        "per_head": per_head,
        "method": "direct-logit-attribution; vocab-space uniform(||1) vs ranking(perp 1) split; "
                  "d*=argmin||M r - 1||; reuses logit_lens_theta_mc.fetch_tensors",
        "ridge": RIDGE, "n_heads": len(pairs),
    }, open(OUT, "w"), indent=1)
    print(f"\n[nvr] wrote {OUT}")


if __name__ == "__main__":
    main()
