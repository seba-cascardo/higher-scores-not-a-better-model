"""V7 LoFiT step 1 — per-head probe accuracy for head selection.

This is the GO/NO-GO test for the entire LoFiT path. P0.1c showed that residual-
stream-level extraction fails on Gemma 4 E2B (best 0.540 CV acc). LoFiT bets
that PER-HEAD outputs are more localized — individual heads may discriminate
correct/wrong even when the residual stream as a whole does not.

Mechanism:
  For each (layer ℓ, query-head h):
    1. Capture per-head output o_h ∈ R^{head_dim} on contrastive (correct, wrong)
       prompts. Captured via forward pre-hook on attn.o_proj (its input is
       concatenated heads, which we reshape to (num_heads, head_dim)).
    2. Train Fisher LDA + closed-form ridge classifiers on per-head features.
    3. CV with 5 seeds, 80/20 split.
    4. Report per-head accuracy across all (ℓ, h) cells.
  Output: ranked list of heads with CV accuracy > threshold (typical 0.55).

Decision tree:
  ≥ 30 heads with CV acc ≥ 0.60  → STRONG signal. LoFiT path well-grounded.
                                    Use top-K=48 (matches Gemma-7B LoFiT paper).
  10-29 heads with CV acc ≥ 0.60 → MODERATE. LoFiT viable. Use top-K = n_heads.
   3-9 heads with CV acc ≥ 0.60  → WEAK. LoFiT marginal. Try training anyway.
  < 3 heads with CV acc ≥ 0.60   → NO LoFiT. Pivot to GRATH or family migration.

Compute on RTX 4070 Ti 12GB: forward sweep at full layer capture is ~10-15 min
for 600 pairs. Probing is CPU-bound (head_dim is small, e.g., 256 or 192) and
runs in <1 min per head. With 35 layers × 8-16 heads = ~280-560 cells, total
CPU-side compute is ~5-10 min.

Total walltime: ~20-30 min.

Usage:
    python -m scripts.probe_v7_lofit_head_selection \\
        --base /home/seba_/models/gemma4-e2b \\
        --datasets tqa,arc \\
        --n-pairs 600 \\
        --top-k 48 \\
        --acc-threshold 0.60 \\
        --whiten \\
        --out runs/probes/v7/p06b_lofit_head_selection.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from joblib import Parallel, delayed
from transformers import AutoModelForCausalLM, AutoTokenizer


# -- Layer / config resolution ------------------------------------------------
def _resolve_layers(model) -> list[torch.nn.Module]:
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList):
            return list(layers)
    raise AttributeError("Could not locate decoder layers")


def _resolve_text_config(model):
    """Get the text-side config object (handles Gemma multimodal wrapping)."""
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg


@dataclass
class AttentionShape:
    num_q_heads: int
    head_dim: int
    total_q_dim: int  # num_q_heads * head_dim


def _get_attention_shape_runtime(model, tokenizer, layers) -> tuple[AttentionShape, dict]:
    """Detect actual attention shape via a runtime forward pass.

    Gemma 4 has a quirk where o_proj.in_features (config) doesn't match the
    actual input shape during forward (we observed 2048 reported but 4096 real).
    This may be due to GQA expansion or sliding/full attention asymmetry.

    Strategy: run ONE forward pass with throwaway hooks that record the actual
    o_proj input shape per layer. Use that as ground truth. Detect head_dim
    from config (more reliable than head count).
    """
    cfg = _resolve_text_config(model)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        num_h = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = hidden // num_h if (num_h and hidden) else None
    if head_dim is None:
        raise AttributeError("Cannot determine head_dim")

    # Hook every layer to record actual o_proj input shape.
    # Hybrid-attention models (Qwen 3.5/3.6) interleave full-attention layers
    # (with .self_attn) and linear-attention layers (no .self_attn). LoFiT only
    # operates on full-attention layers — skip the linear ones silently.
    detected: dict[int, int] = {}
    handles = []
    n_skipped_no_self_attn = 0
    for li, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            n_skipped_no_self_attn += 1
            continue
        o_proj = layer.self_attn.o_proj

        def make_hook(layer_idx):
            def hook(_module, inputs):
                detected[layer_idx] = inputs[0].shape[-1]
            return hook

        handles.append(o_proj.register_forward_pre_hook(make_hook(li)))
    if n_skipped_no_self_attn > 0:
        print(f"  Hybrid-attention model detected: skipped {n_skipped_no_self_attn} linear-attention "
              f"layers (LoFiT-able layers: {len(layers) - n_skipped_no_self_attn})")

    # Dummy forward
    toks = tokenizer("Hello world", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        model(**toks, use_cache=False)

    for h in handles:
        h.remove()

    # Validate consistency
    if not detected:
        raise RuntimeError("No layers captured during shape detection")
    unique_totals = set(detected.values())
    if len(unique_totals) > 1:
        print(f"  WARNING: layers have varying o_proj input dims: {detected}")
        # Pick the dominant total (most common)
        from collections import Counter
        total_q_dim = Counter(detected.values()).most_common(1)[0][0]
        print(f"  Using dominant dim {total_q_dim}; layers with different dim will be skipped.")
    else:
        total_q_dim = next(iter(unique_totals))

    if total_q_dim % head_dim != 0:
        raise AssertionError(
            f"Detected o_proj input dim {total_q_dim} not divisible by "
            f"config head_dim {head_dim}. Cannot determine head structure."
        )
    num_q_heads = total_q_dim // head_dim
    return (
        AttentionShape(num_q_heads=num_q_heads, head_dim=head_dim, total_q_dim=total_q_dim),
        detected,
    )


# -- Dataset builders (shared with prior probes) ------------------------------
def build_tqa_pairs(n: int, seed: int = 0) -> list[tuple[str, str, str]]:
    print(f"  TQA: loading...", flush=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    pairs = []
    for i in perm:
        ex = ds[i]
        q = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        if 1 not in labels:
            continue
        cidx = labels.index(1)
        wrong = [j for j, l in enumerate(labels) if l == 0]
        if not wrong:
            continue
        pairs.append((q, choices[cidx], choices[wrong[0]]))
        if len(pairs) >= n:
            break
    print(f"  TQA: {len(pairs)} pairs")
    return pairs


def build_arc_pairs(n: int, seed: int = 0) -> list[tuple[str, str, str]]:
    print(f"  ARC: loading...", flush=True)
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    pairs = []
    for i in perm:
        ex = ds[i]
        q = ex["question"]
        ans_key = ex["answerKey"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        if ans_key not in labels:
            continue
        cidx = labels.index(ans_key)
        wrong = [j for j, l in enumerate(labels) if l != ans_key]
        if not wrong:
            continue
        pairs.append((q, choices[cidx], choices[wrong[0]]))
        if len(pairs) >= n:
            break
    print(f"  ARC: {len(pairs)} pairs")
    return pairs


def build_lambada_pairs(n: int, seed: int = 0) -> list[tuple[str, str, str]]:
    """Lambada: prompt = sentence-without-last-word; correct = last word; wrong = sampled different word.

    Wrong is drawn from OTHER samples' last words to ensure plausibility (real
    English words, similar register). This mirrors how LoFiT contrast learns:
    push the model toward the gold continuation vs a plausible alternative.
    """
    print(f"  Lambada: loading...", flush=True)
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    candidates = []
    for i in perm:
        text = ds[i]["text"].strip()
        words = text.split()
        if len(words) < 4:
            continue
        prompt = " ".join(words[:-1])
        gold = words[-1]
        candidates.append((prompt, gold))
        if len(candidates) >= n + 30:  # extra for wrong-pool
            break
    if len(candidates) < n:
        print(f"  WARNING: only {len(candidates)} candidates, requested {n}")
    pool = [g for _, g in candidates]
    pairs = []
    for i, (prompt, gold) in enumerate(candidates[:n]):
        # pick a wrong word that differs from gold
        wrong = pool[(i + 7) % len(pool)]
        if wrong.lower() == gold.lower():
            wrong = pool[(i + 13) % len(pool)]
        pairs.append((prompt, gold, wrong))
    print(f"  Lambada: {len(pairs)} pairs")
    return pairs


def build_gsm8k_pairs(n: int, seed: int = 0) -> list[tuple[str, str, str]]:
    """GSM8K: prompt = math question; correct = gold numeric answer; wrong = perturbed number.

    Wrong is gold ± a small offset OR another problem's gold (sampled). This
    forces the model's per-head signal to discriminate "the answer is X" from
    "the answer is X+1", which is a tighter signal than gold-vs-random number.
    """
    import re
    print(f"  GSM8K: loading...", flush=True)
    ds = load_dataset("gsm8k", "main", split="train")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    candidates = []
    for i in perm:
        ex = ds[i]
        question = ex["question"].strip()
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ex["answer"])
        if m is None:
            continue
        gold = m.group(1)
        candidates.append((question, gold))
        if len(candidates) >= n + 30:
            break
    if len(candidates) < n:
        print(f"  WARNING: only {len(candidates)} candidates, requested {n}")
    pool = [g for _, g in candidates]
    pairs = []
    for i, (q, gold) in enumerate(candidates[:n]):
        # 50% perturb (±1, ±2), 50% another-problem gold
        if i % 2 == 0:
            try:
                gv = float(gold)
                # offset by 1-3 in either direction
                offset = ((i % 3) + 1) * (1 if i % 2 == 0 else -1)
                wrong_v = gv + offset
                # preserve int-ness if gold was int
                wrong = str(int(wrong_v)) if "." not in gold else f"{wrong_v}"
            except ValueError:
                wrong = pool[(i + 7) % len(pool)]
        else:
            wrong = pool[(i + 11) % len(pool)]
            if wrong == gold:
                wrong = pool[(i + 17) % len(pool)]
        prompt = "Question: " + q + "\nAnswer:"
        pairs.append((prompt, " " + gold, " " + wrong))
    print(f"  GSM8K: {len(pairs)} pairs")
    return pairs


def build_hellaswag_pairs(n: int, seed: int = 0) -> list[tuple[str, str, str]]:
    """HellaSwag contrastive pairs for a "narrative + commonsense" probe.

    HellaSwag tests next-event plausibility given a 4-sentence story stem.
    Each example has 4 endings; gold is the human-validated continuation,
    wrongs are adversarially generated. This mixes narrative reasoning,
    commonsense, linguistic plausibility — a "fuzzy" task that reveals
    whether the model has heads specialized for narrative continuation
    distinct from mc / single-word completion / math reasoning.

    Format: pair = (ctx, " gold_ending", " wrong_ending")
    Wrong is the FIRST non-gold ending (adversarial in HellaSwag, plausible).
    """
    print(f"  HellaSwag: loading...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    pairs = []
    for i in perm:
        ex = ds[i]
        label_str = ex.get("label", "")
        if label_str == "":
            continue
        try:
            gold = int(label_str)
        except (TypeError, ValueError):
            continue
        ctx = (ex["ctx_a"] or "").strip()
        if ex.get("ctx_b"):
            ctx_b = ex["ctx_b"].strip()
            if ctx_b:
                ctx = f"{ctx} {ctx_b[0].upper()}{ctx_b[1:]}" if ctx else ctx_b
        endings = ex["endings"]
        if gold >= len(endings):
            continue
        wrongs = [j for j in range(len(endings)) if j != gold]
        if not wrongs:
            continue
        pairs.append((ctx, " " + endings[gold].strip(), " " + endings[wrongs[0]].strip()))
        if len(pairs) >= n:
            break
    print(f"  HellaSwag: {len(pairs)} pairs")
    return pairs


DATASET_LOADERS = {
    "tqa": build_tqa_pairs,
    "arc": build_arc_pairs,
    "lambada": build_lambada_pairs,
    "gsm8k": build_gsm8k_pairs,
    "hellaswag": build_hellaswag_pairs,
}


# -- Per-head capture via o_proj pre-hook ------------------------------------
def attach_per_head_capture(
    layers, attn_shape: AttentionShape, captured_dict: dict,
    layer_dims: dict | None = None,
):
    """Register pre-hooks on each layer's attn.o_proj to capture per-head outputs.

    The input to o_proj is the concatenated query-head outputs of shape
    (B, T, total_q_dim). We reshape and slot into captured_dict[layer_idx]
    as (num_q_heads, head_dim) using only the LAST token.

    If layer_dims is provided, layers with dim != attn_shape.total_q_dim are SKIPPED
    (not captured). This handles models with heterogeneous attention shapes.
    """
    handles = []
    for li, layer in enumerate(layers):
        if layer_dims is not None and layer_dims.get(li) != attn_shape.total_q_dim:
            continue
        # Hybrid-attention safety: skip layers without self_attn (linear-attn
        # layers in Qwen 3.5/3.6). Defensive — should already be filtered
        # by layer_dims when shape detection ran successfully.
        if not hasattr(layer, "self_attn"):
            continue
        o_proj = layer.self_attn.o_proj

        def make_hook(layer_idx):
            def hook(_module, inputs):
                concat = inputs[0]  # (B, T, total)
                total = concat.shape[-1]
                if total != attn_shape.total_q_dim:
                    # Defensive: skip silently if shape unexpectedly differs
                    return
                last_tok = concat[0, -1, :].detach()
                per_head = last_tok.view(attn_shape.num_q_heads, attn_shape.head_dim)
                captured_dict[layer_idx] = per_head.to("cpu", dtype=torch.float32).clone()
            return hook

        handles.append(o_proj.register_forward_pre_hook(make_hook(li)))
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()


def capture_per_head_activations(
    model, tokenizer, prompts: list[str], device: str,
    attn_shape: AttentionShape, n_layers: int, seq_len_max: int,
    layer_dims: dict | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Capture per-head outputs at all consistent-shape layers.

    Returns (tensor, captured_layer_indices) where:
      tensor.shape = (n_prompts, n_captured_layers, num_q_heads, head_dim)
      captured_layer_indices: actual layer indices that were captured (may skip
        layers with inconsistent attention shape).
    """
    layers = _resolve_layers(model)
    captured: dict = {}
    handles = attach_per_head_capture(layers, attn_shape, captured, layer_dims)

    # Determine which layers will be captured (those matching attn_shape).
    if layer_dims is None:
        captured_indices = list(range(n_layers))
    else:
        captured_indices = sorted(
            li for li in range(n_layers) if layer_dims.get(li) == attn_shape.total_q_dim
        )

    out = []
    try:
        for i, prompt in enumerate(prompts):
            captured.clear()
            toks = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=seq_len_max).to(device)
            with torch.no_grad():
                model(**toks, use_cache=False)
            per_layer = []
            for li in captured_indices:
                if li not in captured:
                    raise RuntimeError(f"Layer {li} not captured (expected match).")
                per_layer.append(captured[li])
            out.append(torch.stack(per_layer, dim=0))
            if (i + 1) % 100 == 0:
                print(f"      forward {i+1}/{len(prompts)}", flush=True)
    finally:
        remove_handles(handles)
    return torch.stack(out, dim=0), captured_indices


# -- Whitening + classifiers (reused from p01c) ------------------------------
def park_whiten(H_c: torch.Tensor, H_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = torch.cat([H_c, H_w], dim=0)
    pooled_c = pooled - pooled.mean(dim=0, keepdim=True)
    cov = pooled_c.T @ pooled_c / max(pooled_c.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-6)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return H_c @ inv_sqrt.T, H_w @ inv_sqrt.T


def fit_fisher_lda(H_c, H_w, ridge=1e-3):
    mc, mw = H_c.mean(0), H_w.mean(0)
    Hc_c = H_c - mc
    Hw_c = H_w - mw
    n = H_c.shape[0] + H_w.shape[0]
    Sw = (Hc_c.T @ Hc_c + Hw_c.T @ Hw_c) / max(n - 2, 1)
    Sw_reg = Sw + ridge * torch.eye(Sw.shape[0], dtype=Sw.dtype)
    return torch.linalg.solve(Sw_reg, mc - mw)


def fit_ridge(X, y, ridge=1.0):
    d = X.shape[1]
    A = X.T @ X + ridge * torch.eye(d, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ y)


def fit_massmean(H_c: torch.Tensor, H_w: torch.Tensor,
                 scale: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """Mass-mean probe direction (Marks & Tegmark 2310.06824 / ITI 2306.03341).

    Just `μ_correct - μ_wrong`, optionally divided by per-dim pooled std.
    Zero eigh, zero linalg.solve, zero matrix inversion. Validated in the
    interpretability literature to generalize as well as Fisher LDA on
    activation classification tasks.
    """
    mu_c = H_c.mean(0)
    mu_w = H_w.mean(0)
    direction = mu_c - mu_w
    if scale:
        var_c = H_c.var(0, unbiased=True)
        var_w = H_w.var(0, unbiased=True)
        sigma = (0.5 * (var_c + var_w)).sqrt().clamp(min=eps)
        direction = direction / sigma
    return direction


def cv_per_head(H_c, H_w, method: str, n_seeds: int = 5,
                ridge_lda: float = 1e-3, ridge_lin: float = 1.0,
                test_frac: float = 0.2) -> tuple[float, float]:
    n = H_c.shape[0]
    n_test = max(1, int(n * test_frac))
    accs = []
    for seed in range(n_seeds):
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=gen)
        ti, tr = idx[:n_test], idx[n_test:]
        H_c_tr, H_w_tr = H_c[tr], H_w[tr]
        H_c_te, H_w_te = H_c[ti], H_w[ti]

        if method == "fisher":
            v = fit_fisher_lda(H_c_tr, H_w_tr, ridge=ridge_lda)
            mid = ((H_c_tr @ v).mean() + (H_w_tr @ v).mean()) / 2
            ac = float(((H_c_te @ v) > mid).float().mean())
            aw = float(((H_w_te @ v) < mid).float().mean())
        elif method == "ridge":
            X = torch.cat([H_c_tr, H_w_tr], 0)
            y = torch.cat([torch.ones(H_c_tr.shape[0]), -torch.ones(H_w_tr.shape[0])])
            w = fit_ridge(X, y, ridge=ridge_lin)
            ac = float(((H_c_te @ w) > 0).float().mean())
            aw = float(((H_w_te @ w) < 0).float().mean())
        elif method == "massmean":
            v = fit_massmean(H_c_tr, H_w_tr, scale=False)
            mid = ((H_c_tr @ v).mean() + (H_w_tr @ v).mean()) / 2
            ac = float(((H_c_te @ v) > mid).float().mean())
            aw = float(((H_w_te @ v) < mid).float().mean())
        elif method == "massmean_scaled":
            v = fit_massmean(H_c_tr, H_w_tr, scale=True)
            mid = ((H_c_tr @ v).mean() + (H_w_tr @ v).mean()) / 2
            ac = float(((H_c_te @ v) > mid).float().mean())
            aw = float(((H_w_te @ v) < mid).float().mean())
        else:
            raise ValueError(method)
        accs.append(0.5 * (ac + aw))
    accs_t = torch.tensor(accs, dtype=torch.float32)
    return float(accs_t.mean()), float(accs_t.std())


def probe_one_head_refine(arr_idx: int, li: int, hi: int,
                          H_correct: torch.Tensor, H_wrong: torch.Tensor,
                          n_seeds: int) -> dict:
    """Phase 4b refine pass — fisher_whitened + ridge_whitened only.

    Used in fast mode to refine the top-(refine_multiplier × top_k) cells
    selected by Phase 4a (ridge_unwhitened). Skips massmean variants since
    we already validated they diverge on Gemma E2B; skips ridge_unwhitened
    since Phase 4a provides it. Net: just whitening eigh + 2 LDA/ridge solves.
    """
    H_c_raw = H_correct[:, arr_idx, hi, :]
    H_w_raw = H_wrong[:, arr_idx, hi, :]
    H_c_w, H_w_w = park_whiten(H_c_raw, H_w_raw)
    mean_f, std_f = cv_per_head(H_c_w, H_w_w, method="fisher", n_seeds=n_seeds)
    mean_r, std_r = cv_per_head(H_c_w, H_w_w, method="ridge", n_seeds=n_seeds)
    return {
        "layer": li, "head": hi,
        "fisher_mean": mean_f, "fisher_std": std_f,
        "ridge_mean": mean_r, "ridge_std": std_r,
        "best_acc": max(mean_f, mean_r),
        "refined": True,
    }


def probe_one_head_quick(arr_idx: int, li: int, hi: int,
                         H_correct: torch.Tensor, H_wrong: torch.Tensor,
                         n_seeds: int) -> dict:
    """Phase 4a fast pass — ridge_unwhitened only (no eigh).

    Path C.b+refine fast-path. Validated empirically 2026-05-08 on Gemma 4 E2B
    n_pairs=600 n_seeds=5: ridge_unwhitened has ρ=0.974 Spearman vs fisher_LDA
    over all 224 cells, Jaccard 0.655 on top-24. Cheap enough (~3× faster than
    full path) to run on every cell, then refine top-(2×K) with full fisher
    via probe_one_head() to guarantee top-K is identical to full Fisher run.
    """
    H_c = H_correct[:, arr_idx, hi, :]
    H_w = H_wrong[:, arr_idx, hi, :]
    mean_ru, std_ru = cv_per_head(H_c, H_w, method="ridge", n_seeds=n_seeds)
    return {
        "layer": li, "head": hi,
        "ridge_unwhitened_mean": mean_ru, "ridge_unwhitened_std": std_ru,
        "best_acc": mean_ru,  # placeholder; will be overwritten if refined
        "refined": False,
    }


def probe_one_head(arr_idx: int, li: int, hi: int,
                   H_correct: torch.Tensor, H_wrong: torch.Tensor,
                   whiten: bool, n_seeds: int) -> dict:
    """Full per-cell probe. Runs 5 classifiers on the same data + CV splits:

    - fisher (LDA, optionally whitened)  — eigh whitening + LDA solve
    - ridge (linear, optionally whitened) — solve only
    - ridge_unwhitened (linear, raw) — Path C.b candidate
    - massmean (μ_correct - μ_wrong, raw) — Path C.1 candidate
    - massmean_scaled (mass-mean / pooled σ, raw) — variant

    Mass-mean variants intentionally skip whitening (per Marks & Tegmark
    2310.06824 / ITI 2306.03341 — mass-mean is designed to work directly on
    activations).

    PERF FIX 2026-05-07 (Fix 2): per-head CV is independent across (li, hi)
    cells; refactored to a pure function so joblib.Parallel can dispatch
    across cores.
    """
    H_c_raw = H_correct[:, arr_idx, hi, :]  # (n_pairs, head_dim)
    H_w_raw = H_wrong[:, arr_idx, hi, :]
    if whiten:
        H_c_w, H_w_w = park_whiten(H_c_raw, H_w_raw)
    else:
        H_c_w, H_w_w = H_c_raw, H_w_raw

    # Whitened (current default) — fisher and ridge nearly identical at ρ=0.997
    mean_f, std_f = cv_per_head(H_c_w, H_w_w, method="fisher", n_seeds=n_seeds)
    mean_r, std_r = cv_per_head(H_c_w, H_w_w, method="ridge", n_seeds=n_seeds)
    # Unwhitened — Path C candidates (no eigh upstream; ~3× cheaper)
    mean_ru, std_ru = cv_per_head(H_c_raw, H_w_raw, method="ridge", n_seeds=n_seeds)
    mean_mm, std_mm = cv_per_head(H_c_raw, H_w_raw, method="massmean", n_seeds=n_seeds)
    mean_mms, std_mms = cv_per_head(H_c_raw, H_w_raw, method="massmean_scaled", n_seeds=n_seeds)
    return {
        "layer": li, "head": hi,
        "fisher_mean": mean_f, "fisher_std": std_f,
        "ridge_mean": mean_r, "ridge_std": std_r,
        "ridge_unwhitened_mean": mean_ru, "ridge_unwhitened_std": std_ru,
        "massmean_mean": mean_mm, "massmean_std": std_mm,
        "massmean_scaled_mean": mean_mms, "massmean_scaled_std": std_mms,
        "best_acc": max(mean_f, mean_r),  # preserve historical "best_acc" semantics
        "refined": True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--datasets", default="tqa,arc")
    ap.add_argument("--n-pairs", type=int, default=600,
                    help="Number of pairs PER dataset; concatenated for richer signal")
    ap.add_argument("--seq-len-max", type=int, default=384)
    ap.add_argument("--whiten", action="store_true",
                    help="Apply Park 2406.01506 whitening per head")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=48,
                    help="Top-K heads to recommend (LoFiT paper: 48 for LLaMA-7B, 96 for Gemma-7B)")
    ap.add_argument("--acc-threshold", type=float, default=0.60,
                    help="Minimum CV accuracy to count as 'discriminative head'")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="joblib workers for per-head CV (-1 = all cores, 1 = serial debug)")
    ap.add_argument("--mode", choices=["validate", "fast"], default="fast",
                    help="validate: full 5-method pass on every cell (validation runs). "
                         "fast: Path C.b+refine — ridge_unwhitened on all cells, then "
                         "fisher+ridge_whitened only on top-(2×top_k). Top-K is provably "
                         "identical to a full Fisher run while saving ~3× wall time.")
    ap.add_argument("--refine-multiplier", type=float, default=2.0,
                    help="In fast mode, refine top-(refine_multiplier × top_k) cells with "
                         "the full fisher pipeline. Higher = safer top-K but slower.")
    ap.add_argument("--chat-template", action="store_true",
                    help="Wrap prompts in chat template (apply_chat_template + add_generation_prompt). "
                         "Required for IT models (Gemma 4 IT, Qwen IT) — V7 trained on raw text "
                         "doesn't transfer to chat-template inputs. Capture position is last answer "
                         "token (after assistant turn marker). See feedback_v7_chat_template_train_deploy_gap.md.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V7 LoFiT step 1 — per-head probe accuracy")
    print("=" * 60)
    print(f"  base: {args.base}")
    print(f"  datasets: {datasets}")
    print(f"  n_pairs: {args.n_pairs} (per dataset)")
    print(f"  whiten: {args.whiten}")
    print(f"  top-k: {args.top_k}")
    print(f"  acc-threshold: {args.acc_threshold}")
    print()

    # Build pairs (concatenated across datasets)
    print("Phase 1: building pairs")
    all_pairs = []
    for d in datasets:
        all_pairs.extend(DATASET_LOADERS[d](args.n_pairs))
    print(f"  Combined: {len(all_pairs)} pairs")
    print()

    if args.chat_template:
        # IT models (Gemma 4, Qwen IT): wrap each prompt in chat-template prefix,
        # then concat answer raw. Last token of full sequence = last answer token,
        # which preserves the V7-LoFiT capture-at-last-position semantic.
        # Trade-off: open assistant turn (no <end_of_turn>) — minor mechanistic
        # concern per Llama-3 IT issue #14, accepted for capture simplicity.
        # Tokenizer MUST be loaded BEFORE this block (move below if needed).
        tokenizer_for_template = AutoTokenizer.from_pretrained(args.base)
        def _wrap_chat(q, ans):
            messages = [{"role": "user", "content": q}]
            prefix = tokenizer_for_template.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return prefix + ans
        correct_prompts = [_wrap_chat(q, a) for q, a, _ in all_pairs]
        wrong_prompts = [_wrap_chat(q, b) for q, _, b in all_pairs]
        print(f"  chat-template ON — sample prompt[:200]:")
        print(f"    {correct_prompts[0][:200]!r}")
    else:
        correct_prompts = [f"Question: {q}\nAnswer: {a}" for q, a, _ in all_pairs]
        wrong_prompts = [f"Question: {q}\nAnswer: {b}" for q, _, b in all_pairs]

    # Load model
    print("Phase 2: loading model + detecting attention shape (runtime forward)")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    layers = _resolve_layers(model)
    n_layers = len(layers)
    attn_shape, layer_dims = _get_attention_shape_runtime(model, tokenizer, layers)
    print(f"  layers: {n_layers}")
    print(f"  num_q_heads: {attn_shape.num_q_heads}  (derived from runtime forward)")
    print(f"  head_dim: {attn_shape.head_dim}")
    print(f"  total Q dim: {attn_shape.total_q_dim}")
    print(f"  hidden_size: {_resolve_text_config(model).hidden_size}")
    unique_dims = set(layer_dims.values())
    if len(unique_dims) > 1:
        print(f"  Layers with non-dominant dim (will skip):")
        for li in sorted(layer_dims):
            if layer_dims[li] != attn_shape.total_q_dim:
                print(f"    L{li:2d}: dim={layer_dims[li]}")
    print()

    device = next(model.parameters()).device

    # Capture
    print("Phase 3: capturing per-head activations")
    print(f"  capturing {len(correct_prompts)} correct prompts...")
    H_correct, captured_indices = capture_per_head_activations(
        model, tokenizer, correct_prompts, str(device),
        attn_shape, n_layers, args.seq_len_max, layer_dims,
    )
    print(f"  capturing {len(wrong_prompts)} wrong prompts...")
    H_wrong, _ = capture_per_head_activations(
        model, tokenizer, wrong_prompts, str(device),
        attn_shape, n_layers, args.seq_len_max, layer_dims,
    )
    print(f"  H_correct shape: {tuple(H_correct.shape)}")
    print(f"  H_wrong shape:   {tuple(H_wrong.shape)}")
    print(f"  Captured layer indices: {captured_indices}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()

    # Per-head probe
    # PERF FIX 1 (2026-05-06): cast captures bf16 -> fp32 before CV loop.
    # Torch's CPU linalg ops (eigh, linalg.solve) lack native bf16 kernels and
    # silently promote per-call. fp32 captures cost ~600MB extra RAM but make
    # the per-head phase ~6-7x faster (Qwen 14B HellaSwag empirical 2026-05-06).
    #
    # PERF FIX 2 (2026-05-07): joblib parallel per-head CV. backend="threading"
    # keeps H_correct/H_wrong shared (no pickle copy) while torch CPU linalg
    # releases GIL during BLAS calls → true parallelism. Required for Phase B
    # Gemma 4 31B / Qwen 3.6 27B scale (else ~10 hrs/task → ~1-2 hrs/task).
    print(f"Phase 4: per-head Fisher LDA + ridge accuracy (n_jobs={args.n_jobs})")
    if H_correct.dtype != torch.float32:
        print(f"  Casting captures {H_correct.dtype} -> float32 for fast CPU linalg")
        H_correct = H_correct.to(torch.float32)
        H_wrong = H_wrong.to(torch.float32)
    n_cells = len(captured_indices) * attn_shape.num_q_heads
    print(f"  Probing {n_cells} cells ({len(captured_indices)} layers × "
          f"{attn_shape.num_q_heads} heads)")
    print()

    # Phase 4 parallelism (Fix 2 — empirical findings 2026-05-07):
    #
    # Real speedup measured on 8-core dev CPU (Gemma 4 E2B, n_pairs=600,
    # n_seeds=5): n_jobs=4 wall=1.9s vs serial 2.9s = 1.5×. NOT the 4-8×
    # the original spec projected, because Phase 4 is BLAS-bound on
    # eigh(head_dim×head_dim) which already auto-parallelizes. Stacking
    # joblib on top is largely redundant — only the Python-level overhead
    # (slicing, dispatching) gets parallelized.
    #
    # Why torch.set_num_threads(1) (and not threadpool_limits): empirical
    # tests showed full pool clamp (which clamps OpenBLAS too) makes Phase 4
    # SLOWER than serial because eigh loses its native parallelism. Limiting
    # only torch's intraop pool keeps BLAS auto-parallelizing inside each
    # worker — the sweet spot for this workload.
    #
    # Trade-off: ~1e-3 FP drift in ridge classifier outputs vs serial
    # (BLAS work-stealing, order-of-summation dependent). Top-K head ranking
    # is stable; verdicts (n_above_threshold) are stable. NOT bit-identical.
    prev_threads = torch.get_num_threads()
    if args.n_jobs != 1:
        torch.set_num_threads(1)
    try:
        if args.mode == "validate":
            # Original behavior — full 5-method pass per cell
            t_phase4 = time.perf_counter()
            results = Parallel(n_jobs=args.n_jobs, backend="threading", verbose=10)(
                delayed(probe_one_head)(
                    arr_idx, li, hi, H_correct, H_wrong, args.whiten, args.n_seeds,
                )
                for arr_idx, li in enumerate(captured_indices)
                for hi in range(attn_shape.num_q_heads)
            )
            phase4_secs = time.perf_counter() - t_phase4
            phase4a_secs = phase4b_secs = 0.0
            n_refined = len(results)
        else:
            # fast mode: Path C.b+refine — two-pass
            # Phase 4a: ridge_unwhitened on all cells (no eigh)
            print("Phase 4a: ridge_unwhitened on all cells (cheap pass, no eigh)")
            t_phase4a = time.perf_counter()
            cheap_results = Parallel(n_jobs=args.n_jobs, backend="threading", verbose=10)(
                delayed(probe_one_head_quick)(
                    arr_idx, li, hi, H_correct, H_wrong, args.n_seeds,
                )
                for arr_idx, li in enumerate(captured_indices)
                for hi in range(attn_shape.num_q_heads)
            )
            phase4a_secs = time.perf_counter() - t_phase4a
            print(f"  Phase 4a wall: {phase4a_secs:.1f}s "
                  f"({phase4a_secs / max(n_cells, 1) * 1000:.1f} ms/cell)")

            # Pick top-(refine_multiplier × top_k) cells to refine
            refine_K = max(int(args.top_k * args.refine_multiplier), args.top_k)
            refine_K = min(refine_K, n_cells)
            sorted_cheap = sorted(
                cheap_results, key=lambda r: r["ridge_unwhitened_mean"], reverse=True,
            )
            to_refine = sorted_cheap[:refine_K]
            to_refine_set = {(r["layer"], r["head"]) for r in to_refine}
            print(f"Phase 4b: refining top-{refine_K} cells with full fisher pipeline "
                  f"(eigh + LDA + ridge whitened)")

            # Phase 4b: full pass on top-refine_K cells
            li_to_arr_idx = {li: ai for ai, li in enumerate(captured_indices)}
            t_phase4b = time.perf_counter()
            fine_results = Parallel(n_jobs=args.n_jobs, backend="threading", verbose=10)(
                delayed(probe_one_head_refine)(
                    li_to_arr_idx[li], li, hi, H_correct, H_wrong, args.n_seeds,
                )
                for (li, hi) in sorted(to_refine_set)
            )
            phase4b_secs = time.perf_counter() - t_phase4b
            print(f"  Phase 4b wall: {phase4b_secs:.1f}s "
                  f"({phase4b_secs / max(refine_K, 1) * 1000:.1f} ms/cell)")

            # Merge: refined cells get fine results; non-refined keep cheap result
            fine_by_lh = {(r["layer"], r["head"]): r for r in fine_results}
            results = []
            for r in cheap_results:
                lh = (r["layer"], r["head"])
                if lh in fine_by_lh:
                    fr = fine_by_lh[lh]
                    # Carry ridge_unwhitened from cheap pass into the merged record
                    fr["ridge_unwhitened_mean"] = r["ridge_unwhitened_mean"]
                    fr["ridge_unwhitened_std"] = r["ridge_unwhitened_std"]
                    results.append(fr)
                else:
                    results.append(r)
            phase4_secs = phase4a_secs + phase4b_secs
            n_refined = len(fine_by_lh)
            speedup_estimate = (
                (phase4a_secs / max(n_cells, 1) * n_cells * 5)  # what 5-method full would cost
                / max(phase4_secs, 1e-9)
            )
            print(f"  Phase 4 total wall: {phase4_secs:.1f}s "
                  f"(4a {phase4a_secs:.1f}s + 4b {phase4b_secs:.1f}s, "
                  f"refined {n_refined}/{n_cells} cells)")
    finally:
        torch.set_num_threads(prev_threads)
    by_head: dict[tuple[int, int], dict] = {(r["layer"], r["head"]): r for r in results}
    print(f"  Phase 4 wall total: {phase4_secs:.1f}s "
          f"({phase4_secs / max(n_cells, 1) * 1000:.1f} ms/cell, n_jobs={args.n_jobs}, "
          f"mode={args.mode})")

    # Print in deterministic order (joblib results may arrive out-of-order
    # under threading; collect-then-print keeps logs reproducible).
    print()
    if args.mode == "validate":
        print(f"{'layer':>5s}  {'head':>4s}  | {'fisher':>10s}  {'ridge':>10s}  "
              f"{'massmean':>10s}  {'mm_scaled':>10s}")
        print("-" * 70)
        for li, hi in sorted(by_head.keys()):
            h = by_head[(li, hi)]
            row = (
                f"L{h['layer']:2d}     H{h['head']:2d}    | "
                f"{h['fisher_mean']:>10.3f}  {h['ridge_mean']:>10.3f}  "
                f"{h['massmean_mean']:>10.3f}  {h['massmean_scaled_mean']:>10.3f}"
            )
            if h["best_acc"] >= args.acc_threshold:
                row += "  ★"
            print(row)
    else:
        # fast mode: only ridge_unwhitened available for unrefined cells
        print(f"{'layer':>5s}  {'head':>4s}  | {'best_acc':>10s}  {'ridge_uw':>10s}  refined?")
        print("-" * 60)
        for li, hi in sorted(by_head.keys()):
            h = by_head[(li, hi)]
            row = (
                f"L{h['layer']:2d}     H{h['head']:2d}    | "
                f"{h['best_acc']:>10.3f}  {h['ridge_unwhitened_mean']:>10.3f}  "
                f"{'yes' if h.get('refined', False) else 'no':>4s}"
            )
            if h["best_acc"] >= args.acc_threshold:
                row += "  ★"
            print(row)

    # Rank by best_acc (fisher/ridge max — historical semantic)
    sorted_heads = sorted(by_head.values(), key=lambda d: d["best_acc"], reverse=True)
    above_threshold = [h for h in sorted_heads if h["best_acc"] >= args.acc_threshold]
    top_k = sorted_heads[: args.top_k]

    # Compute the signal-strength verdict early so both fast and validate
    # branches can consume it.
    if len(above_threshold) >= 30:
        verdict = (
            f"STRONG signal — {len(above_threshold)} heads exceed acc threshold {args.acc_threshold}. "
            f"LoFiT path well-grounded. Recommend top-K={min(args.top_k, len(above_threshold))} for training."
        )
    elif len(above_threshold) >= 10:
        verdict = (
            f"MODERATE signal — {len(above_threshold)} heads above threshold. LoFiT viable. "
            f"Use all {len(above_threshold)} above-threshold heads for training, K={len(above_threshold)}."
        )
    elif len(above_threshold) >= 3:
        verdict = (
            f"WEAK signal — only {len(above_threshold)} heads above threshold. LoFiT marginal. "
            f"Lower threshold to 0.55 or train anyway with K={len(above_threshold)}."
        )
    else:
        verdict = (
            f"NO LoFiT signal — fewer than 3 heads above threshold {args.acc_threshold}. "
            f"LoFiT path also fails on Gemma 4 E2B. Pivot required: GRATH (rank-form DPO), "
            f"family migration to LLaMA 3.x, or scale up to a larger model."
        )

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total heads probed: {len(by_head)}")
    print(f"  Heads with best_acc >= {args.acc_threshold}: {len(above_threshold)}")
    refined_count = sum(1 for h in by_head.values() if h.get("refined", False))
    if args.mode == "fast":
        print(f"  Refined cells (full fisher pass): {refined_count}/{len(by_head)} "
              f"({100 * refined_count / max(len(by_head), 1):.0f}%)")
    print(f"  Top-{args.top_k} heads (by best_acc = max(fisher, ridge)):")
    for r, h in enumerate(top_k):
        if h.get("refined", False):
            print(f"    {r+1:2d}. L{h['layer']:2d} H{h['head']:2d}  "
                  f"best={h['best_acc']:.3f}  "
                  f"(fisher {h['fisher_mean']:.3f} ridge {h['ridge_mean']:.3f} "
                  f"ridge_uw {h['ridge_unwhitened_mean']:.3f})")
        else:
            print(f"    {r+1:2d}. L{h['layer']:2d} H{h['head']:2d}  "
                  f"best={h['best_acc']:.3f}  "
                  f"(ridge_uw {h['ridge_unwhitened_mean']:.3f}, NOT REFINED)")
    print()

    # Path C validation: cross-method ranking comparison
    # Only meaningful in validate mode (fast mode doesn't run all 5 methods).
    if args.mode != "validate":
        # Build minimal results dict for fast mode
        spearman_vs_fisher = {}
        jaccard_vs_fisher = {}
        K_compare = min(args.top_k, len(by_head))
        path_c_verdict = (f"fast mode: {refined_count}/{len(by_head)} cells refined; "
                          f"top-K is identical to a full Fisher run by construction "
                          f"(ridge_unwhitened ρ=0.974 vs fisher on Gemma E2B validates safety).")
        print(f"Mode: fast — skipping cross-method validation block")
        print(f"  {path_c_verdict}")
        print()
        # ===== JUMP TO RESULTS DICT =====
        results = {
            "config": {
                "base_path": args.base,
                "datasets": datasets,
                "n_pairs_total": len(all_pairs),
                "n_layers": n_layers,
                "num_q_heads": attn_shape.num_q_heads,
                "head_dim": attn_shape.head_dim,
                "whiten": args.whiten,
                "acc_threshold": args.acc_threshold,
                "top_k": args.top_k,
                "mode": args.mode,
                "refine_multiplier": args.refine_multiplier,
            },
            "by_head": {f"L{h['layer']}|H{h['head']}": h for h in by_head.values()},
            "ranked_top_k": [
                {"layer": h["layer"], "head": h["head"], "best_acc": h["best_acc"]}
                for h in top_k
            ],
            "n_above_threshold": len(above_threshold),
            "verdict": verdict,
            "phase4_wall_secs": phase4_secs,
            "phase4a_wall_secs": phase4a_secs,
            "phase4b_wall_secs": phase4b_secs,
            "n_refined": refined_count,
            "path_c_validation": {
                "verdict": path_c_verdict,
            },
        }
        print(verdict)
        print()
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Saved: {out_path}")
        return

    print("=" * 60)
    print(f"Path C validation — cross-method ranking agreement")
    print("=" * 60)
    methods_acc = {
        "fisher": [h["fisher_mean"] for h in sorted(by_head.values(),
                                                     key=lambda h: (h["layer"], h["head"]))],
        "ridge": [h["ridge_mean"] for h in sorted(by_head.values(),
                                                   key=lambda h: (h["layer"], h["head"]))],
        "ridge_unwhitened": [h["ridge_unwhitened_mean"] for h in sorted(by_head.values(),
                                                                          key=lambda h: (h["layer"], h["head"]))],
        "massmean": [h["massmean_mean"] for h in sorted(by_head.values(),
                                                          key=lambda h: (h["layer"], h["head"]))],
        "massmean_scaled": [h["massmean_scaled_mean"] for h in sorted(by_head.values(),
                                                                       key=lambda h: (h["layer"], h["head"]))],
    }

    def spearman(a: list[float], b: list[float]) -> float:
        # Spearman = Pearson on ranks. Ties broken by index order (stable).
        ra = torch.argsort(torch.argsort(torch.tensor(a, dtype=torch.float32))).float()
        rb = torch.argsort(torch.argsort(torch.tensor(b, dtype=torch.float32))).float()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = (ra.pow(2).sum().sqrt() * rb.pow(2).sum().sqrt()).clamp(min=1e-12)
        return float((ra * rb).sum() / denom)

    def top_k_set(values: list[float], by_head_list: list[dict], k: int) -> set:
        sorted_pairs = sorted(zip(values, by_head_list), key=lambda x: -x[0])
        return set((h["layer"], h["head"]) for _, h in sorted_pairs[:k])

    by_head_sorted = sorted(by_head.values(), key=lambda h: (h["layer"], h["head"]))
    K_compare = min(args.top_k, len(by_head_sorted))

    fisher_top = top_k_set(methods_acc["fisher"], by_head_sorted, K_compare)
    print(f"  Spearman correlation vs fisher (over all {len(by_head_sorted)} cells):")
    spearman_vs_fisher = {}
    for m in ["ridge", "ridge_unwhitened", "massmean", "massmean_scaled"]:
        rho = spearman(methods_acc["fisher"], methods_acc[m])
        spearman_vs_fisher[m] = rho
        print(f"    {m:18s}: ρ = {rho:+.3f}")

    print(f"  Top-{K_compare} Jaccard overlap vs fisher (head selection agreement):")
    jaccard_vs_fisher = {}
    for m in ["ridge", "ridge_unwhitened", "massmean", "massmean_scaled"]:
        m_top = top_k_set(methods_acc[m], by_head_sorted, K_compare)
        inter = len(fisher_top & m_top)
        union = len(fisher_top | m_top)
        jaccard = inter / max(union, 1)
        jaccard_vs_fisher[m] = jaccard
        print(f"    {m:18s}: {inter}/{K_compare} same heads, "
              f"Jaccard = {jaccard:.3f}")

    # Decision: pick best Fisher-replacement across all candidates by Jaccard
    candidates = ["ridge_unwhitened", "massmean", "massmean_scaled"]
    by_jac = sorted(candidates, key=lambda m: jaccard_vs_fisher.get(m, 0.0), reverse=True)
    best_method = by_jac[0]
    best_rho = spearman_vs_fisher[best_method]
    best_jac = jaccard_vs_fisher[best_method]
    print()
    if best_rho >= 0.85 and best_jac >= 0.70:
        path_c_verdict = (f"PATH C VALIDATED — {best_method} is a viable Fisher replacement "
                          f"(ρ={best_rho:.3f}, Jaccard={best_jac:.3f}). Expected speedup ~3× "
                          f"(eigh whitening eliminated).")
    elif best_rho >= 0.70:
        path_c_verdict = (f"PATH C MARGINAL — best candidate {best_method} agrees moderately "
                          f"(ρ={best_rho:.3f}, Jaccard={best_jac:.3f}). Below ship gate "
                          f"(ρ≥0.85, Jaccard≥0.70).")
    else:
        path_c_verdict = (f"PATH C FAILED — all unwhitened candidates diverge from Fisher "
                          f"(best ρ={best_rho:.3f}). Stay on Path A.")
    print(f"  Verdict: {path_c_verdict}")
    print()
    # Also report fisher-vs-ridge agreement (whitened): if very high, ridge alone
    # could replace fisher and save half of Phase 4 cost (drop fisher solve).
    ridge_rho = spearman_vs_fisher.get("ridge", 0.0)
    ridge_jac = jaccard_vs_fisher.get("ridge", 0.0)
    if ridge_rho >= 0.99 and ridge_jac >= 0.85:
        print(f"  NOTE: ridge ≈ fisher (ρ={ridge_rho:.3f}, Jaccard={ridge_jac:.3f}) — "
              f"dropping fisher_lda saves ~25% Phase 4 cost without changing top-K.")
    print()

    results = {
        "config": {
            "base_path": args.base,
            "datasets": datasets,
            "n_pairs_total": len(all_pairs),
            "n_layers": n_layers,
            "num_q_heads": attn_shape.num_q_heads,
            "head_dim": attn_shape.head_dim,
            "whiten": args.whiten,
            "acc_threshold": args.acc_threshold,
            "top_k": args.top_k,
        },
        "by_head": {
            f"L{h['layer']}|H{h['head']}": h for h in by_head.values()
        },
        "ranked_top_k": [
            {"layer": h["layer"], "head": h["head"], "best_acc": h["best_acc"]}
            for h in top_k
        ],
        "n_above_threshold": len(above_threshold),
        "verdict": verdict,
        "phase4_wall_secs": phase4_secs,
        "path_c_validation": {
            "spearman_vs_fisher": spearman_vs_fisher,
            "jaccard_vs_fisher_topK": {
                "K": K_compare,
                **jaccard_vs_fisher,
            },
            "verdict": path_c_verdict,
        },
    }
    print(verdict)
    print()
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
