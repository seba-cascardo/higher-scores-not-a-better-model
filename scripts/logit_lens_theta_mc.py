"""Logit-lens / direct-logit-attribution of the V7-mc adapter offset (theta_mc).

WHY: W_know + the arbiter showed the +35 MC-scoring lift is OFF-AXIS to the
correctness direction. The open reviewer question is "off-axis to correctness, but
on-axis to WHAT?". This reads out what the per-head trained offset directly
promotes/suppresses in the vocabulary — converting the negative ('not correctness')
into a positive named mechanism (e.g. option-letters / positions / high-freq tokens).

METHOD (direct logit attribution, the DIRECT-path approximation):
  For adapter head (L,h): the offset alpha[i]*theta[i] (head_dim) is added to the
  o_proj INPUT at head h's slice. Its residual-stream write is
      r = o_proj.weight[L][:, h*hd:(h+1)*hd] @ (alpha*theta)
  Its DIRECT logit effect (skipping layers L+1..final, the logit-lens approximation)
  is   delta_logits = W_U @ ( (1 + norm_weight) * r )            (Gemma RMSNorm gain)
  W_U = tied embed_tokens.weight. Ranking of delta_logits is scale/softcap-invariant.
  Per-head accuracy of the approximation grows with L (L55 ~exact, L30 looser).

WEIGHTS: only embed_tokens + final norm + o_proj[adapter layers] are needed (~3GB),
fetched by safetensors range-read from google/gemma-4-31b-it (gated; uses your hf
login) and cached to runs/logit_lens/g31_tensors.pt. No GPU, no full model load.

Usage (local CPU):  python scripts/logit_lens_theta_mc.py
"""
import json
import os

import torch
from transformers import AutoTokenizer

REPO = "google/gemma-4-31b-it"
SHARD = "model-00001-of-00002.safetensors"
MC = "runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt"
CACHE = "runs/logit_lens/g31_tensors.pt"
OUT = "runs/logit_lens/theta_mc_logit_lens.json"
TOPK = 30

_ST_DTYPE = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
             "F64": torch.float64}


def fetch_tensors(layers):
    """Range-read embed + final norm + o_proj[layers] from the HF shard (~3GB)."""
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if os.path.exists(CACHE):
        print(f"[ll] using cached tensors {CACHE}")
        return torch.load(CACHE, weights_only=False, map_location="cpu")
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    path = f"{REPO}/{SHARD}"
    print(f"[ll] opening {path} (range-read header)...", flush=True)
    want = {"embed": "model.language_model.embed_tokens.weight",
            "norm": "model.language_model.norm.weight"}
    for L in layers:
        want[f"o_proj_{L}"] = f"model.language_model.layers.{L}.self_attn.o_proj.weight"
    out = {}
    with fs.open(path, "rb") as f:
        hsize = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(hsize))
        data0 = 8 + hsize
        for alias, name in want.items():
            meta = header[name]
            b, e = meta["data_offsets"]
            f.seek(data0 + b)
            raw = f.read(e - b)
            t = torch.frombuffer(bytearray(raw), dtype=_ST_DTYPE[meta["dtype"]])
            out[alias] = t.reshape(meta["shape"]).clone()
            print(f"[ll]   {alias}: {tuple(out[alias].shape)} {meta['dtype']} "
                  f"({(e-b)/1e9:.2f} GB)", flush=True)
    torch.save(out, CACHE)
    print(f"[ll] cached -> {CACHE}")
    return out


def categorize(tok_str):
    s = tok_str
    core = s.strip().lstrip("▁")  # gemma uses U+2581 for leading space
    if core == "":
        return "whitespace"
    if len(core) == 1 and core.isalpha():
        return "single_letter"
    if core.isdigit():
        return "digit"
    if all(not c.isalnum() for c in core):
        return "punct/symbol"
    if core.isalpha() and core[0].isupper():
        return "Capitalized_word"
    if core.isalpha():
        return "word"
    return "mixed/subword"


def main():
    mc = torch.load(MC, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in p) for p in mc["layer_head_pairs"]]
    alpha = mc["alpha"].to(torch.float32)
    theta = mc["theta"].to(torch.float32)           # [n, head_dim]
    hd = theta.shape[1]
    layers = sorted(set(L for L, _ in pairs))
    print(f"[ll] {len(pairs)} adapter heads, head_dim={hd}, layers={layers}")

    T = fetch_tensors(layers)
    embed = T["embed"].to(torch.float32)            # [vocab, d_model] = W_U (tied)
    gain = (1.0 + T["norm"].to(torch.float32))      # Gemma RMSNorm: (1+w)
    vocab, d_model = embed.shape
    oproj = {L: T[f"o_proj_{L}"].to(torch.float32) for L in layers}  # [d_model, total_q]
    print(f"[ll] vocab={vocab} d_model={d_model}")

    tok = AutoTokenizer.from_pretrained(REPO)

    def topk_tokens(delta, k=TOPK):
        vp, ip = torch.topk(delta, k)
        vn, intn = torch.topk(-delta, k)
        prom = [(tok.convert_ids_to_tokens(int(i)), round(float(v), 3)) for i, v in zip(ip, vp)]
        supp = [(tok.convert_ids_to_tokens(int(i)), round(float(-v), 3)) for i, v in zip(intn, vn)]
        return prom, supp

    agg = torch.zeros(d_model, dtype=torch.float32)
    per_head = []
    from collections import Counter
    for i, (L, h) in enumerate(pairs):
        W = oproj[L]
        total_q = W.shape[1]
        nh = total_q // hd
        if h >= nh or total_q % hd != 0:
            print(f"[ll]   SKIP L{L}h{h}: total_q={total_q} hd={hd} nh={nh}")
            continue
        off = alpha[i] * theta[i]                   # [hd]
        r = W[:, h * hd:(h + 1) * hd] @ off          # [d_model] residual write
        agg += r
        delta = embed @ (gain * r)                   # [vocab] direct logit effect
        prom, _ = topk_tokens(delta, 10)
        per_head.append({"layer": L, "head": h, "resid_norm": round(float(r.norm()), 3),
                         "top_promoted": prom})

    # aggregate (net direct write of all 48 heads)
    agg_delta = embed @ (gain * agg)
    agg_prom, agg_supp = topk_tokens(agg_delta, TOPK)
    cats = Counter(categorize(t) for t, _ in agg_prom)

    print("\n" + "=" * 64)
    print("AGGREGATE (net direct-logit write of all 48 heads)")
    print("=" * 64)
    print("PROMOTED (top %d):" % TOPK)
    for t, v in agg_prom:
        print(f"   {v:+8.2f}  {t!r:20s}  [{categorize(t)}]")
    print("\nSUPPRESSED (top %d):" % TOPK)
    for t, v in agg_supp[:15]:
        print(f"   {v:+8.2f}  {t!r:20s}  [{categorize(t)}]")
    print("\nCATEGORY MIX of top-%d promoted:" % TOPK)
    for c, n in cats.most_common():
        print(f"   {n:2d}  {c}")

    print("\nPER-HEAD top-3 promoted (first 12 heads):")
    for ph in per_head[:12]:
        t3 = ", ".join(f"{t!r}" for t, _ in ph["top_promoted"][:3])
        print(f"   L{ph['layer']:2d}h{ph['head']:2d} (|r|={ph['resid_norm']:.1f}): {t3}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"aggregate_promoted": agg_prom, "aggregate_suppressed": agg_supp,
               "category_mix": dict(cats), "per_head": per_head,
               "method": "direct logit attribution (direct-path approx), Gemma (1+w) norm gain",
               "n_heads": len(pairs)}, open(OUT, "w"), indent=1)
    print(f"\n[ll] wrote {OUT}")
    print("READ: a coherent promoted set (letters/digits/positions/high-freq) => the off-axis "
          "direction has a NAMED mechanism. Diffuse/incoherent => the direct-path approx is "
          "weak for these (mostly early) layers; the offset acts through downstream nonlinearity.")


if __name__ == "__main__":
    main()
