"""nnsight CAUSAL patch of the V7-mc offset (theta_mc) on REAL MC prompts — scoring face.

WHY (pendiente del logit-lens, handoff 2026-06-23 §1): logit_lens_theta_mc.py read out
theta_mc's lexical signature with a LINEARIZED direct-logit-attribution (fixed RMSNorm gain,
direct path only, no inter-layer reprocessing). It found NO clean lexical mechanism; the direct
signature was "suppress structural tokens (Capitalized/punct)" ~ the EOS/formatting-collapse
face. The open question: does that DLA signature SURVIVE the real non-linear downstream
reprocessing?  This confirms it CAUSALLY — runs the real model on real scoring prompts and
measures the ACTUAL output-logit delta base->adapter, then (1) compares it to the DLA delta and
(2) localizes which adapted layers carry the effect.

TWO measurements (user 2026-07-06: "ambas", scoring only — generation/entropy deferred to P-dose):
  1. LEXICAL SIGNATURE (causal):   aggregate real delta over scoring positions -> top-k promoted/
     suppressed tokens + category-mix (reuses logit_lens categorize) + cos(delta_causal, delta_DLA).
     A high cos => the DLA lexical signature is causally real (survives reprocessing); a low cos =>
     the offset's real effect is dominated by downstream non-linearity the DLA cannot see.
  2. LOCALIZATION (causal):        patch ONE adapted layer at a time -> per-layer delta; rank layers
     by ||delta_L|| and cos(delta_L, delta_full). Answers "where does the scoring effect live?".

WHY nnsight (not the existing install_lofit_hooks): for the SCORING face the manual hooks already
suffice — BUT running nnsight here on the target, with the equivalence SANITY below
(nnsight-patch == install_lofit_hooks, cos~1 on the delta), VALIDATES the nnsight+Gemma-4 chain on
a cheap, verifiable scoring experiment BEFORE we depend on it for the entropy-collapse-during-
generation instrumentation (P-dose, where nnsight's .generate() intervention is the killer feature
the hooks lack). So this script de-risks that future use as a side effect of producing the numbers.

PATCH MECHANICS (exact equivalence to install_lofit_hooks): the hooks add alpha*theta to the
per-head slice of o_proj's INPUT. o_proj is a bias-free Linear, so o_proj(x+off)=o_proj(x)+off@W^T.
Thus adding the constant r_layer = offset_flat @ o_proj.weight^T to o_proj's OUTPUT is
mathematically identical (up to bf16 rounding) and avoids nnsight's version-dependent `.input`
tuple ambiguity. The SANITY block asserts cos(delta_nnsight, delta_hooks) ~ 1.

DLA delta (linearized reference, from logit_lens_theta_mc): delta_DLA = W_U @ ((1+norm_w) * r_agg),
r_agg = sum over adapted layers of r_layer. Computed CPU fp32, matching the logit-lens method.

NOTE: Stage-A canonical repro should have PASSED in the same pod (guards BASE/offsets).

Usage (pod, GPU; cd /workspace/MSAP per block; needs `pip install nnsight`):
  BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' \
        2>/dev/null | grep -v '/.locks/' | head -1)")
  python scripts/patch_theta_mc_nnsight_pod.py --base "$BASE" \
      --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --task arc_challenge --limit 400 --loc-limit 80 --apply-chat-template \
      --out runs/mechanism_v8/theta_mc_nnsight_arc.json
  # scale=1.0 (baked strength). --offset-scale multiplies alpha (same semantics as run_lm_eval_v7).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.score_mc_premise_pod import load_items, build_context, PREMISE, CORPUS
from scripts.run_lm_eval_v7 import _load_model_multimodal, _filter_compatible_layers
from scripts.eval_with_router import (
    install_lofit_hooks, remove_handles, _resolve_layers, _get_attention_shape,
)
from scripts.patch_theta_mc_causal_pod import _find_final_norm
from scripts.logit_lens_theta_mc import categorize
from transformers import AutoTokenizer

TOPK = 30


# ---------------------------------------------------------------------------- helpers
def _resolve_layers_path(model) -> str:
    """Return the dotted attribute path to the decoder-layer ModuleList (mirror of
    eval_with_router._resolve_layers, but returning the PATH string so we can re-walk it
    over the nnsight Envoy tree inside a trace)."""
    candidates = [
        "model.language_model.layers",
        "model.layers",
        "language_model.layers",
        "layers",
    ]
    for path in candidates:
        node = model
        ok = True
        for p in path.split("."):
            if hasattr(node, p):
                node = getattr(node, p)
            else:
                ok = False
                break
        if ok and isinstance(node, torch.nn.ModuleList):
            return path
    raise AttributeError("Could not locate decoder layers path for nnsight walk")


def _walk(root, path):
    node = root
    for p in path.split("."):
        node = getattr(node, p)
    return node


def _compute_r_layers(hf_model, layers, lhp, alpha, theta, attn_shape):
    """r_layer[L] (d_model,) fp32 = (sum over that layer's adapted heads of alpha*theta placed in
    the head slice) @ o_proj.weight^T. Equivalent residual write of the offset at layer L.
    Returns (dict L->r_fp32_cpu, r_agg_fp32_cpu)."""
    num_q, hd = attn_shape.num_q_heads, attn_shape.head_dim
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for idx, (li, hi) in enumerate(lhp):
        by_layer.setdefault(int(li), []).append((int(hi), idx))
    r_layers, d_model = {}, None
    for L, items in by_layer.items():
        o_proj = layers[L].self_attn.o_proj
        W = o_proj.weight.detach().float().cpu()          # (d_model, num_q*hd)
        d_model = W.shape[0]
        off_flat = torch.zeros(num_q * hd, dtype=torch.float32)
        for hi, idx in items:
            off_flat[hi * hd:(hi + 1) * hd] = (alpha[idx].float() * theta[idx].float())
        r_layers[L] = W @ off_flat                        # (d_model,) fp32 cpu
    r_agg = torch.zeros(d_model, dtype=torch.float32)
    for r in r_layers.values():
        r_agg += r
    return r_layers, r_agg


def _dla_delta(hf_model, r_agg_cpu):
    """Linearized direct-logit delta (logit-lens method): W_U @ ((1+norm_w) * r_agg), CPU fp32.
    Returns delta_DLA (vocab,) fp32 cpu."""
    W_U = hf_model.get_input_embeddings().weight.detach().float().cpu()   # tied unembed (vocab,d)
    fn, _ = _find_final_norm(hf_model)
    gain = (1.0 + fn.weight.detach().float().cpu()) if fn is not None else torch.ones(W_U.shape[1])
    return W_U @ (gain * r_agg_cpu)                        # (vocab,) fp32


def _scoring_positions(tokenizer, cfg, q, opts, gold, apply_ct, device):
    """(ids (1,T), positions list) that PREDICT the gold-option continuation tokens — where the
    +35 acc_norm lift lives (mirror of patch_theta_mc_causal_pod --positions option)."""
    ctx = build_context(tokenizer, cfg["prefix"] + cfg["cond"].format(q=q), apply_ct)
    ctx_ids = tokenizer(ctx, add_special_tokens=False).input_ids
    full = tokenizer(ctx + " " + opts[gold], add_special_tokens=False).input_ids
    n_ctx = len(ctx_ids)
    pos = list(range(n_ctx - 1, len(full) - 1)) or [len(full) - 1]
    ids = torch.tensor([full], device=device)
    return ids, pos


def _topk(delta, tokenizer, k=TOPK):
    vp, ip = torch.topk(delta, k)
    vn, intn = torch.topk(-delta, k)
    prom = [(tokenizer.convert_ids_to_tokens(int(i)), round(float(v), 3)) for i, v in zip(ip, vp)]
    supp = [(tokenizer.convert_ids_to_tokens(int(i)), round(float(-v), 3)) for i, v in zip(intn, vn)]
    return prom, supp


def _cos(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--task", required=True, choices=list(PREMISE))
    ap.add_argument("--limit", type=int, default=400, help="items for the lexical-signature aggregate")
    ap.add_argument("--loc-limit", type=int, default=80, help="items for the per-layer localization sweep")
    ap.add_argument("--offset-scale", type=float, default=1.0,
                    help="multiply alpha (same semantics as run_lm_eval_v7); 1.0=baked strength, 0=base")
    ap.add_argument("--sanity-n", type=int, default=5, help="items for the nnsight==hooks equivalence check")
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not (args.base or "").strip():
        raise SystemExit("FATAL: --base empty (fresh pod shell loses $BASE). Re-set it.")

    print("=" * 78, flush=True)
    print(f"nnsight causal patch | task={args.task} limit={args.limit} loc_limit={args.loc_limit} "
          f"scale={args.offset_scale} chat_template={args.apply_chat_template}", flush=True)
    print("=" * 78, flush=True)

    # -- import nnsight late so the missing-dep error is actionable -----------------
    try:
        from nnsight import LanguageModel
    except Exception as e:
        raise SystemExit(f"FATAL: nnsight import failed ({e}). In the pod: pip install nnsight") from e

    cfg = PREMISE[args.task]
    items = load_items(args.task, max(args.limit, args.loc_limit))
    print(f"  {len(items)} items from {CORPUS[args.task]}", flush=True)

    torch_dtype = getattr(torch, args.dtype)
    print(f"  loading base {args.base} ...", flush=True)
    t0 = time.time()
    hf_model = _load_model_multimodal(args.base, torch_dtype)
    hf_model.eval()
    hf_model.requires_grad_(False)   # inference-only: no autograd graph (nnsight would else OOM)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = hf_model.device
    print(f"  loaded in {time.time()-t0:.0f}s  device={device}", flush=True)

    # -- offsets + compatible-layer filter + scale ---------------------------------
    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    lhp, alpha, theta = blob["layer_head_pairs"], blob["alpha"], blob["theta"]
    layers = _resolve_layers(hf_model)
    attn_shape = _get_attention_shape(hf_model, layers)
    lhp, alpha, theta = _filter_compatible_layers(hf_model, tokenizer, attn_shape, lhp, alpha, theta)
    alpha = alpha.float() * float(args.offset_scale)      # scale semantics == run_lm_eval_v7
    adapted_layers = sorted({int(L) for L, _ in lhp})
    print(f"  {len(lhp)} adapted heads over {len(adapted_layers)} layers: {adapted_layers}", flush=True)

    # -- residual-write vectors + DLA reference ------------------------------------
    r_layers_cpu, r_agg_cpu = _compute_r_layers(hf_model, layers, lhp, alpha, theta, attn_shape)
    r_layers = {L: r.to(device=device, dtype=torch_dtype) for L, r in r_layers_cpu.items()}
    print(f"  building DLA reference (W_U @ (1+norm)*r_agg, CPU fp32) ...", flush=True)
    delta_dla = _dla_delta(hf_model, r_agg_cpu)           # (vocab,) fp32 cpu
    dla_prom, dla_supp = _topk(delta_dla, tokenizer, args.topk)
    print(f"    DLA top promoted: {[t for t,_ in dla_prom[:6]]}", flush=True)

    # -- nnsight wrapper (reuse the already-loaded, tested HF model) ---------------
    lm = LanguageModel(hf_model, tokenizer=tokenizer)
    layers_path = _resolve_layers_path(hf_model)
    print(f"  nnsight wrapped; decoder-layers path = '{layers_path}'", flush=True)

    def _base_logits(ids, pos):
        with torch.no_grad():
            z = hf_model(ids, use_cache=False).logits[0, pos].float().cpu()
        return z                                          # (P, vocab) fp32 cpu

    def _nnsight_patched_logits(ids, pos, patch_layers):
        """Real forward with r_layer added to o_proj.output for each L in patch_layers.
        torch.no_grad(): nnsight builds the autograd graph by default -> retaining the 60-layer
        activations across traces OOMs on the 34GB free after the 62GB weights. Pure inference."""
        with torch.no_grad(), lm.trace(ids, scan=False, validate=False):
            for L in patch_layers:
                o_proj = _walk(lm, layers_path)[L].self_attn.o_proj
                cur = o_proj.output                       # capture pre-patch proxy first
                o_proj.output[:] = cur + r_layers[L]       # constant residual write (o_proj linear)
            saved = lm.output.logits[0, pos].save()
        return saved.float().cpu()                        # (P, vocab) fp32 cpu

    # === SANITY: nnsight output-patch == install_lofit_hooks (delta cos ~ 1) ======
    print("\n" + "-" * 78, flush=True)
    print(f"  SANITY equivalence (nnsight vs install_lofit_hooks) on {args.sanity_n} items ...", flush=True)
    cos_eq = []
    for (did, q, opts, gold) in items[:args.sanity_n]:
        ids, pos = _scoring_positions(tokenizer, cfg, q, opts, gold, args.apply_chat_template, device)
        zb = _base_logits(ids, pos)
        # hooks path
        h = install_lofit_hooks(layers, alpha, theta, lhp, attn_shape)
        try:
            with torch.no_grad():
                z_hooks = hf_model(ids, use_cache=False).logits[0, pos].float().cpu()
        finally:
            remove_handles(h)
        # nnsight path (all adapted layers)
        z_ns = _nnsight_patched_logits(ids, pos, adapted_layers)
        cos_eq.append(_cos(z_ns - zb, z_hooks - zb))
    cos_eq_mean = sum(cos_eq) / max(len(cos_eq), 1)
    print(f"    mean cos(delta_nnsight, delta_hooks) = {cos_eq_mean:.5f}  (per-item {[round(c,4) for c in cos_eq]})",
          flush=True)
    if cos_eq_mean < 0.99:
        print("    ⚠ WARNING: equivalence BELOW 0.99 — nnsight patch may not mirror the hooks. "
              "Do NOT trust downstream numbers; investigate before P-dose gen instrumentation.", flush=True)
    else:
        print("    ✓ nnsight+Gemma-4 chain VALIDATED (patch mirrors the baked adapter). "
              "De-risks the P-dose generation-entropy instrumentation.", flush=True)

    # === MEASUREMENT 1: causal lexical signature (aggregate over scoring positions) ==
    print("\n" + "-" * 78, flush=True)
    print(f"  M1 lexical signature: real delta over scoring positions, {min(args.limit,len(items))} items ...",
          flush=True)
    vocab = delta_dla.shape[0]
    dla_norm = float(delta_dla.norm())
    delta_sum = torch.zeros(vocab, dtype=torch.float64)   # accumulate real delta (vocab,)
    cos_pos_sum = 0.0                                     # accumulate per-position cos vs DLA
    n_pos_total = 0
    t0 = time.time()
    sig_items = items[:args.limit]
    for i, (did, q, opts, gold) in enumerate(sig_items):
        ids, pos = _scoring_positions(tokenizer, cfg, q, opts, gold, args.apply_chat_template, device)
        zb = _base_logits(ids, pos)                       # (P, vocab)
        za = _nnsight_patched_logits(ids, pos, adapted_layers)
        d = (za - zb)                                     # (P, vocab) fp32 cpu
        # per-position cos vs the fixed DLA vector: distinguishes 'DLA is misleading' (low here too)
        # from 'effect is context-dependent so the AGGREGATE cos washes out' (high here, low aggregate).
        cp = (d @ delta_dla) / (d.norm(dim=1) * dla_norm + 1e-12)   # (P,)
        cos_pos_sum += float(cp.sum())
        delta_sum += d.double().sum(0)
        n_pos_total += d.shape[0]
        if (i + 1) % 25 == 0 or (i + 1) == len(sig_items):
            el = time.time() - t0
            print(f"    [{i+1}/{len(sig_items)}] {el:.0f}s  {(i+1)/max(el,1e-9):.2f} it/s", flush=True)
    delta_causal = (delta_sum / max(n_pos_total, 1)).float()   # mean real delta (vocab,)
    cos_per_pos = cos_pos_sum / max(n_pos_total, 1)            # mean per-position cos vs DLA
    caus_prom, caus_supp = _topk(delta_causal, tokenizer, args.topk)
    from collections import Counter
    cat_prom = Counter(categorize(t) for t, _ in caus_prom)
    cat_supp = Counter(categorize(t) for t, _ in caus_supp)
    cos_dla = _cos(delta_causal, delta_dla)
    print(f"    cos(delta_causal, delta_DLA) = {cos_dla:.4f} (aggregate)  |  "
          f"per-position mean = {cos_per_pos:.4f}", flush=True)
    print(f"    read: both low => DLA misleading; per-pos HIGH & aggregate low => effect is "
          f"context-dependent (aggregate washes signed context components)", flush=True)
    print(f"    causal PROMOTED top-8: {[t for t,_ in caus_prom[:8]]}", flush=True)
    print(f"    causal SUPPRESSED top-8: {[t for t,_ in caus_supp[:8]]}", flush=True)
    print(f"    promoted category-mix: {dict(cat_prom.most_common())}", flush=True)
    print(f"    suppressed category-mix: {dict(cat_supp.most_common())}", flush=True)

    # === MEASUREMENT 2: causal localization (one adapted layer at a time) =========
    print("\n" + "-" * 78, flush=True)
    print(f"  M2 localization: per-layer patch over {min(args.loc_limit,len(items))} items "
          f"({len(adapted_layers)} layers) ...", flush=True)
    loc_items = items[:args.loc_limit]
    per_layer_sum = {L: torch.zeros(vocab, dtype=torch.float64) for L in adapted_layers}
    full_sum = torch.zeros(vocab, dtype=torch.float64)
    loc_pos_total = 0
    t0 = time.time()
    for i, (did, q, opts, gold) in enumerate(loc_items):
        ids, pos = _scoring_positions(tokenizer, cfg, q, opts, gold, args.apply_chat_template, device)
        zb = _base_logits(ids, pos)
        za_full = _nnsight_patched_logits(ids, pos, adapted_layers)
        full_sum += (za_full - zb).double().sum(0)
        for L in adapted_layers:
            za_L = _nnsight_patched_logits(ids, pos, [L])
            per_layer_sum[L] += (za_L - zb).double().sum(0)
        loc_pos_total += za_full.shape[0]
        if (i + 1) % 10 == 0 or (i + 1) == len(loc_items):
            el = time.time() - t0
            print(f"    [{i+1}/{len(loc_items)}] {el:.0f}s  {(i+1)/max(el,1e-9):.2f} it/s", flush=True)
    delta_full_loc = (full_sum / max(loc_pos_total, 1)).float()
    loc = []
    for L in adapted_layers:
        dL = (per_layer_sum[L] / max(loc_pos_total, 1)).float()
        loc.append({"layer": L,
                    "delta_norm": round(float(dL.norm()), 4),
                    "cos_vs_full": round(_cos(dL, delta_full_loc), 4),
                    "cos_vs_dla": round(_cos(dL, delta_dla), 4)})
    tot_norm = sum(x["delta_norm"] for x in loc) or 1.0
    for x in loc:
        x["norm_share"] = round(x["delta_norm"] / tot_norm, 4)
    loc.sort(key=lambda x: -x["delta_norm"])
    print(f"    top-5 layers by ||delta_L||: "
          f"{[(x['layer'], x['delta_norm'], x['norm_share']) for x in loc[:5]]}", flush=True)
    print(f"    (norm_share is a rough additive proxy; the effect is non-linear, see caveat in JSON)",
          flush=True)

    # -- write ---------------------------------------------------------------------
    payload = {
        "task": args.task, "positions": "option", "offset_scale": args.offset_scale,
        "apply_chat_template": args.apply_chat_template, "n_adapted_heads": len(lhp),
        "adapted_layers": adapted_layers,
        "sanity_equivalence": {"nnsight_vs_hooks_cos_mean": cos_eq_mean, "per_item": cos_eq,
                               "note": "cos of (adapter-base) logit delta; >=0.99 => nnsight patch "
                                       "mirrors install_lofit_hooks (o_proj linear identity)."},
        "lexical_signature": {
            "n_items": len(sig_items), "n_positions": n_pos_total,
            "cos_causal_vs_dla": cos_dla, "cos_per_position_vs_dla": cos_per_pos,
            "causal_promoted": caus_prom, "causal_suppressed": caus_supp,
            "causal_promoted_category_mix": dict(cat_prom), "causal_suppressed_category_mix": dict(cat_supp),
            "dla_promoted": dla_prom, "dla_suppressed": dla_supp,
            "read": "cos_causal_vs_dla is the AGGREGATE (mean delta over items vs fixed DLA vector); "
                    "cos_per_position_vs_dla is the mean of per-position cos. BOTH low => DLA is "
                    "causally misleading (non-linear reprocessing dominates). Per-position HIGH but "
                    "aggregate low => the effect is context-dependent and the aggregate washes out "
                    "signed context components (DLA still locally partially faithful).",
        },
        "localization": {
            "n_items": len(loc_items), "n_positions": loc_pos_total, "by_layer": loc,
            "caveat": "delta_L = patch layer L alone; norm_share is a rough proxy (the full effect is "
                      "NOT the linear sum of per-layer deltas). cos_vs_full ranks alignment with the "
                      "aggregate effect; delta_norm ranks marginal magnitude.",
        },
        "method": "nnsight real-forward output-patch (r_layer = offset_flat @ o_proj.weight^T added to "
                  "o_proj.output; identical to install_lofit_hooks input-add for a linear o_proj). "
                  "DLA reference = W_U @ ((1+norm)*r_agg) CPU fp32 (logit-lens method).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 78, flush=True)
    print(f"  saved: {args.out}", flush=True)
    print(f"  VERDICT INPUTS: sanity_cos={cos_eq_mean:.4f} | cos_causal_vs_dla={cos_dla:.4f} | "
          f"top_layer={loc[0]['layer'] if loc else None}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
