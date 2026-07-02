"""L11 — token-frequency control for theta_mc (audit 2026-07-02 §3-B12, ledger B5g).

B12 refuted the *temperature* (uniform-shift) leg of a Stolfo confidence mechanism
(cos(agg, d*)=0.023, ranking-dominant). The *other* Stolfo leg is token-FREQUENCY: a
logit shift proportional to token log-frequency. This derives the frequency direction
d_freq = argmin_r ||M r - f||  (M = W_U * gain, f = unigram log-freq) and reports
cos(theta_mc_agg, d_freq).  The B5g artifact never existed (ledger confirms exhaustive
2026-06-24 search); the draft forbids citing the ~0.02 number until re-derived — this
re-derives it.

Reuses logit_lens_theta_mc.fetch_tensors (cached g31_tensors.pt: W_U + final norm +
o_proj) and the B12 aggregate-write construction — NO full-model load, CPU-local.

Gate (plan L11-Step3):
  |cos(theta_mc_agg, d_freq)| <~ 3/sqrt(d_model) ~ 0.041  -> the "not token-frequency"
    claim is re-derived and citable in §5.6.
  otherwise -> the claim is RETIRED from §5.6 (numbers-first).
Identification check: corr(M @ d_freq, f) must be high (r>0.9) or d_freq is not the
frequency direction and the cos is meaningless -> abort.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logit_lens_theta_mc import fetch_tensors, MC  # reuse cached-tensor loader


def build_agg(mc, T):
    """Aggregate 48-head net residual write of theta_mc in d_model space (== B12 `agg`)."""
    pairs = [tuple(int(x) for x in p) for p in mc["layer_head_pairs"]]
    alpha = mc["alpha"].to(torch.float32)
    theta = mc["theta"].to(torch.float32)
    hd = theta.shape[1]
    layers = sorted(set(L for L, _ in pairs))
    oproj = {L: T[f"o_proj_{L}"].to(torch.float32) for L in layers}
    d_model = T["embed"].shape[1]
    agg = torch.zeros(d_model, dtype=torch.float32)
    for i, (L, h) in enumerate(pairs):
        W = oproj[L]
        nh = W.shape[1] // hd
        if h >= nh or W.shape[1] % hd != 0:
            continue
        off = alpha[i] * theta[i]
        agg += W[:, h * hd:(h + 1) * hd] @ off
    return agg


def build_logfreq(vocab, tokenizer, token_budget, progress_every=20000):
    """Unigram log-frequency over the tokenizer vocab from wikitext-103 (streaming, +1 smoothed)."""
    from datasets import load_dataset
    counts = np.zeros(vocab, dtype=np.float64)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    seen_tokens = 0
    n_docs = 0
    for ex in ds:
        txt = ex["text"]
        if not txt or not txt.strip():
            continue
        ids = tokenizer.encode(txt, add_special_tokens=False)
        for t in ids:
            if 0 <= t < vocab:
                counts[t] += 1.0
        seen_tokens += len(ids)
        n_docs += 1
        if n_docs % progress_every == 0:
            print(f"    [freq] {n_docs} docs, {seen_tokens/1e6:.2f}M tokens", flush=True)
        if seen_tokens >= token_budget:
            break
    print(f"    [freq] done: {n_docs} docs, {seen_tokens/1e6:.2f}M tokens, "
          f"{int((counts>0).sum())}/{vocab} vocab seen", flush=True)
    f = np.log(counts + 1.0)  # +1 smoothing then log
    return f.astype(np.float32), seen_tokens, int((counts > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-budget", type=int, default=5_000_000)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", type=Path, default=Path("runs/oq1_functional_axis/freq_control.json"))
    args = ap.parse_args()

    print("[freq-ctrl] loading offsets + cached tensors ...", flush=True)
    mc = torch.load(MC, weights_only=False, map_location="cpu")
    layers = sorted(set(int(p[0]) for p in mc["layer_head_pairs"]))
    T = fetch_tensors(layers)
    embed = T["embed"].to(torch.float32)          # [vocab, d_model] = W_U (tied)
    gvec = (1.0 + T["norm"].to(torch.float32))    # Gemma RMSNorm gain (1+w)
    vocab, d_model = embed.shape
    print(f"[freq-ctrl] vocab={vocab} d_model={d_model}", flush=True)

    agg = build_agg(mc, T)
    agg_unit = agg / agg.norm()
    print(f"[freq-ctrl] |agg|={float(agg.norm()):.2f}", flush=True)

    print("[freq-ctrl] loading Gemma 4 tokenizer ...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31b-it")
    print("[freq-ctrl] building unigram log-freq from wikitext-103 ...", flush=True)
    f_np, seen_tokens, vocab_seen = build_logfreq(vocab, tok, args.token_budget)
    f = torch.from_numpy(f_np)

    # d_freq = argmin_r ||M r - f||, M = embed * gvec.  Normal equations (== B12 d* but RHS=f):
    #   (M^T M + ridge I) x = M^T f ;  M^T M = (embed^T embed) ⊙ (gvec gvec^T) ;  M^T f = gvec ⊙ (embed^T f)
    print("[freq-ctrl] Gram (embed^T embed) ...", flush=True)
    gram = embed.T @ embed
    AtA = gram * gvec.unsqueeze(0) * gvec.unsqueeze(1)
    AtA += args.ridge * torch.eye(d_model)
    Atb = gvec * (embed.T @ f)
    print("[freq-ctrl] solving for d_freq ...", flush=True)
    d_freq = torch.linalg.solve(AtA, Atb)
    d_freq = d_freq / d_freq.norm()

    # identification: does d_freq's logit image reproduce f?
    img = embed @ (gvec * d_freq)
    r_ident = float(np.corrcoef(img.numpy(), f.numpy())[0, 1])
    # also correlation with mean-centered f (frequency *ranking*, orthogonal to uniform)
    fc = f - f.mean()
    img_c = img - img.mean()
    r_ident_ranking = float(np.corrcoef(img_c.numpy(), fc.numpy())[0, 1])

    cos_agg_dfreq = float(torch.dot(agg_unit, d_freq))
    # a mean-centered variant: frequency-ranking direction only
    Atb_c = gvec * (embed.T @ fc)
    d_freq_rank = torch.linalg.solve(AtA, Atb_c)
    d_freq_rank = d_freq_rank / d_freq_rank.norm()
    cos_agg_dfreq_rank = float(torch.dot(agg_unit, d_freq_rank))

    null = 1.0 / math.sqrt(d_model)
    gate_thresh = 3.0 * null
    within = abs(cos_agg_dfreq) <= gate_thresh
    ident_ok = r_ident > 0.9 or r_ident_ranking > 0.9
    print(f"[freq-ctrl] identification r(M d_freq, f)={r_ident:.4f} "
          f"(ranking r={r_ident_ranking:.4f}); ident_ok={ident_ok}", flush=True)
    print(f"[freq-ctrl] cos(agg, d_freq)={cos_agg_dfreq:+.4f} "
          f"(ranking-only {cos_agg_dfreq_rank:+.4f}); null 1/sqrt(d)={null:.4f}, "
          f"3x-null gate={gate_thresh:.4f}", flush=True)

    if not ident_ok:
        gate = (f"IDENTIFICATION FAIL: r(M d_freq, f)={r_ident:.3f} < 0.9 — d_freq is not the "
                f"frequency direction; cos is meaningless. Escalate to P8c (better basis).")
    elif within:
        gate = (f"RE-DERIVED: |cos(agg, d_freq)|={abs(cos_agg_dfreq):.4f} <= 3/sqrt(d)={gate_thresh:.4f} "
                f"-> theta_mc is NOT the token-frequency component of a Stolfo mechanism either; "
                f"the §5.6 claim is citable.")
    else:
        gate = (f"RETIRE: |cos(agg, d_freq)|={abs(cos_agg_dfreq):.4f} > 3/sqrt(d)={gate_thresh:.4f} "
                f"-> theta_mc has a frequency component; RETIRE the 'not token-frequency' claim.")
    print(f"[gate] {gate}", flush=True)

    out = {
        "method": ("d_freq = argmin_r ||M r - f||, M = W_U * (1+norm), f = unigram log-freq "
                   "(wikitext-103 train, +1 smoothed); reuses logit_lens_theta_mc.fetch_tensors "
                   "and the B12 aggregate-write construction."),
        "vocab": int(vocab), "d_model": int(d_model),
        "wikitext_tokens_seen": int(seen_tokens), "vocab_seen": vocab_seen,
        "ridge": args.ridge,
        "cos_agg_dfreq": cos_agg_dfreq, "cos_agg_dfreq_ranking_only": cos_agg_dfreq_rank,
        "ident_corr_M_dfreq_vs_f": r_ident, "ident_corr_ranking": r_ident_ranking,
        "null_1_over_sqrt_d": null, "gate_threshold_3x_null": gate_thresh,
        "b12_reference_cos_agg_dstar": 0.0231,
        "gate_verdict": gate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
