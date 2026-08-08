"""OQ1-funcional (E2) — extract per-head functional directions v_inf / v_ret.

Pre-reg: docs/protocols/2026-06-22-oq1-vg-execution-prereg.md §2.

GOAL
----
On Gemma 4 31B IT, extract, in ONE aligned run, the per-(layer, head) functional
direction that discriminates correct-vs-wrong answers in TWO task families:

  v_inf[L, h]  = massmean direction for an INFERENCE-form family (ARC + HellaSwag)
  v_ret[L, h]  = massmean direction for a  RETRIEVAL-form family (KR factual MC:
                 SciQ + MedMCQA; graceful fallback to TruthfulQA if JSONL absent)

The downstream local probe (scripts/probe_oq1_functional_angle.py, separate file)
measures cos(v_inf, v_ret) both RAW and WHITENED (Mahalanobis under the base
within-class covariance) to test whether the two functional axes are orthogonal
(=> §5 "same heads carry orthogonal directions" survives; process-LoFiT GATE = GREEN),
aligned (=> rewrite §5), or anti-colinear (=> single axis; GATE = RED).

WHY ONE RUN
-----------
v_inf, v_ret and theta_mc must all live on the SAME (layer, head) grid. The grid
is fixed by `capture_per_head_activations` -> `captured_indices` (which layers were
captured under the dominant attention shape). Extracting both families in one
process guarantees identical `captured_indices`, `attn_shape` and `layer_dims`, so
tuple (layer, head) is a sound join key across v_inf / v_ret / theta_mc.

GEMMA 4 31B GOTCHAS (respected here, see also the imported helpers)
-------------------------------------------------------------------
- Heterogeneous head_dim (sliding-attn 256 / full-attn 512): we use
  `_get_attention_shape_runtime` + `layer_dims` so `capture_per_head_activations`
  SKIPS layers whose o_proj input dim != the dominant total. Never assume uniform.
- IT model => chat-template MUST be applied consistently with how V7 was trained
  (apply_chat_template + add_generation_prompt, answer appended raw, capture at the
  LAST token). Mirrors probe_v7_lofit_head_selection.py lines ~804-824.

REUSE (imported, NOT reimplemented), from probe_v7_lofit_head_selection.py:
  DATASET_LOADERS, _resolve_layers, _resolve_text_config,
  _get_attention_shape_runtime, capture_per_head_activations,
  fit_massmean, cv_per_head.

COVARIANCE STORAGE DECISION (documented, see field `covariance_storage` in out)
-------------------------------------------------------------------------------
The whitened-cos downstream needs, per (L, h), the pooled within-class covariance
of the BASE per-head activations so the local probe can Mahalanobis-whiten. A full
head_dim x head_dim cov for EVERY captured cell is ~1GB/family at head_dim 256
(too large). Per the pre-reg ("for ~48 selected heads it's fine") we store the full
pooled covariance (head_dim x head_dim, fp32) ONLY for ELIGIBLE heads — those with
massmean CV probe-accuracy >= --elig-acc in BOTH families (the heads the angle test
will actually use). For every captured cell we additionally store the compact
per-class means (mu_correct, mu_wrong) for both families, which is cheap and lets
the probe recompute the raw direction / sanity-check alignment without the model.
Pooled cov is computed per family separately (cov_inf, cov_ret) since the two
families have different activation distributions; the probe can average or pick.

OUTPUT (.pt dict, schema_version 'oq1-func-1.0') — see the saved `state` below.

Usage (pod, GPU):
    cd /workspace/MSAP && pwd
    python -m scripts.extract_functional_axis_perhead \
        --base /workspace/models/gemma4-31b-it \
        --inference-datasets arc,hellaswag \
        --retrieval-datasets phase1c_kr_sciq,phase1c_kr_medmcqa \
        --n-pairs 600 \
        --seq-len-max 384 \
        --chat-template \
        --out runs/oq1_functional_axis/functional_directions.pt
"""
from __future__ import annotations

import argparse
import gc
import os
import datetime as dt
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the canonical V7 head-probe machinery. Importing (not reimplementing)
# guarantees byte-identical capture semantics, shape detection and direction
# fitting vs the rest of the LoFiT pipeline.
from scripts.probe_v7_lofit_head_selection import (
    DATASET_LOADERS,
    _resolve_layers,
    _resolve_text_config,
    _get_attention_shape_runtime,
    capture_per_head_activations,
    fit_massmean,
    cv_per_head,
)


SCHEMA_VERSION = "oq1-func-1.0"
MC_OFFSETS_PATH = os.environ.get("MC_OFFSETS", "runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt")


def _build_prompts(all_pairs, chat_template: bool, tokenizer):
    """Form (correct_prompts, wrong_prompts) EXACTLY like probe_v7_lofit_head_selection.

    chat_template ON (IT models): apply_chat_template(user=q, add_generation_prompt)
    then concat the answer raw; capture position is the last answer token.
    chat_template OFF: 'Question: {q}\\nAnswer: {a}'.
    """
    if chat_template:
        def _wrap_chat(q, ans):
            messages = [{"role": "user", "content": q}]
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return prefix + ans
        correct_prompts = [_wrap_chat(q, a) for q, a, _ in all_pairs]
        wrong_prompts = [_wrap_chat(q, b) for q, _, b in all_pairs]
    else:
        correct_prompts = [f"Question: {q}\nAnswer: {a}" for q, a, _ in all_pairs]
        wrong_prompts = [f"Question: {q}\nAnswer: {b}" for q, _, b in all_pairs]
    return correct_prompts, wrong_prompts


def _load_family_pairs(dataset_keys, n_pairs):
    """Load + concatenate contrastive pairs for one family.

    Returns (pairs, used_keys). KR loaders may raise FileNotFoundError when the
    JSONL corpus is absent on the pod; the CALLER handles the retrieval fallback.
    Any single bad key here propagates so the caller can decide.
    """
    pairs = []
    used = []
    for d in dataset_keys:
        if d not in DATASET_LOADERS:
            raise KeyError(
                f"Unknown dataset key '{d}'. Available: {sorted(DATASET_LOADERS)}"
            )
        loaded = DATASET_LOADERS[d](n_pairs)
        pairs.extend(loaded)
        used.append(d)
    return pairs, used


def _load_retrieval_pairs(retrieval_keys, fallback_keys, n_pairs):
    """Retrieval family with graceful fallback.

    Tries `retrieval_keys` (default KR SciQ + MedMCQA). If ANY of them raises
    FileNotFoundError (JSONL absent on pod), prints a clear WARNING and falls
    back to `fallback_keys` (default TruthfulQA mc1, a HF dataset). Records
    which keys were actually used + whether the fallback fired.
    """
    try:
        pairs, used = _load_family_pairs(retrieval_keys, n_pairs)
        return pairs, used, False
    except FileNotFoundError as e:
        print("=" * 60, flush=True)
        print(f"  WARNING: retrieval JSONL not found ({e}).", flush=True)
        print(f"  Falling back retrieval family {retrieval_keys} -> {fallback_keys}.", flush=True)
        print(f"  CAVEAT: TruthfulQA is retrieval-ish/inference-ish — treat the "
              f"resulting angle as a SENSITIVITY-CHECK, not primary.", flush=True)
        print("=" * 60, flush=True)
        pairs, used = _load_family_pairs(fallback_keys, n_pairs)
        return pairs, used, True


def _massmean_directions_and_means(H_correct, H_wrong, captured_indices, num_q_heads):
    """Per (arr_idx, head): v = massmean dir, plus per-class means for storage.

    Returns:
      v        : Tensor[n_captured_layers, num_q_heads, head_dim]  (mu_c - mu_w)
      mu_c     : Tensor[n_captured_layers, num_q_heads, head_dim]
      mu_w     : Tensor[n_captured_layers, num_q_heads, head_dim]
    H_* are fp32 of shape (n_pairs, n_captured_layers, num_q_heads, head_dim).
    """
    n_lay = len(captured_indices)
    head_dim = H_correct.shape[-1]
    v = torch.zeros(n_lay, num_q_heads, head_dim, dtype=torch.float32)
    mu_c = torch.zeros(n_lay, num_q_heads, head_dim, dtype=torch.float32)
    mu_w = torch.zeros(n_lay, num_q_heads, head_dim, dtype=torch.float32)
    total = n_lay * num_q_heads
    done = 0
    t0 = time.time()
    for ai in range(n_lay):
        for hi in range(num_q_heads):
            H_c = H_correct[:, ai, hi, :]
            H_w = H_wrong[:, ai, hi, :]
            v[ai, hi] = fit_massmean(H_c, H_w, scale=False)
            mu_c[ai, hi] = H_c.mean(0)
            mu_w[ai, hi] = H_w.mean(0)
            done += 1
        if (ai + 1) % 5 == 0 or (ai + 1) == n_lay:
            print(f"      massmean dirs [{done}/{total}] "
                  f"(layer-block {ai+1}/{n_lay}, {time.time()-t0:.0f}s)", flush=True)
    return v, mu_c, mu_w


def _cv_acc_per_head(H_correct, H_wrong, captured_indices, num_q_heads, n_seeds):
    """Per (arr_idx, head) massmean CV accuracy (so the probe can filter heads).

    Returns Tensor[n_captured_layers, num_q_heads] of CV mean accuracy. Uses the
    same `cv_per_head(..., method='massmean')` the V7 probe uses — keeps the
    eligibility threshold (acc>=0.60) comparable to the rest of the repo.
    """
    n_lay = len(captured_indices)
    acc = torch.zeros(n_lay, num_q_heads, dtype=torch.float32)
    total = n_lay * num_q_heads
    done = 0
    t0 = time.time()
    for ai in range(n_lay):
        for hi in range(num_q_heads):
            mean_acc, _ = cv_per_head(
                H_correct[:, ai, hi, :], H_wrong[:, ai, hi, :],
                method="massmean", n_seeds=n_seeds,
            )
            acc[ai, hi] = mean_acc
            done += 1
        if (ai + 1) % 5 == 0 or (ai + 1) == n_lay:
            print(f"      cv-acc [{done}/{total}] "
                  f"(layer-block {ai+1}/{n_lay}, {time.time()-t0:.0f}s)", flush=True)
    return acc


def _pooled_cov_for_cells(H_correct, H_wrong, cells):
    """Pooled WITHIN-CLASS covariance for a given set of (arr_idx, head) cells.

    Mirrors park_whiten's covariance estimate: pool correct+wrong, center on the
    POOLED mean, cov = X_c^T X_c / (N-1). (park_whiten centers on the pooled mean,
    so we replicate that exactly — the probe whitens with eigh(cov).)

    cells: iterable of (arr_idx, head) tuples.
    Returns dict[(arr_idx, head)] -> Tensor[head_dim, head_dim] fp32.
    """
    out = {}
    cells = list(cells)
    t0 = time.time()
    for n_done, (ai, hi) in enumerate(cells, 1):
        H_c = H_correct[:, ai, hi, :]
        H_w = H_wrong[:, ai, hi, :]
        pooled = torch.cat([H_c, H_w], dim=0)
        pooled_c = pooled - pooled.mean(dim=0, keepdim=True)
        cov = pooled_c.T @ pooled_c / max(pooled_c.shape[0] - 1, 1)
        out[(ai, hi)] = cov.to(torch.float32).clone()
        if n_done % 25 == 0 or n_done == len(cells):
            print(f"      pooled cov [{n_done}/{len(cells)}] ({time.time()-t0:.0f}s)",
                  flush=True)
    return out


def _load_theta_mc(captured_indices, num_q_heads, head_dim):
    """Load mc-adapter offsets and align theta_mc to (arr_idx, head) cells.

    The mc offsets blob (runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt) stores
    `layer_head_pairs: list[(layer, head)]`, `alpha: [n]`, `theta: [n, head_dim]`.
    The APPLIED per-head offset at inference is `alpha[i] * theta[i]` (verified in
    install_lofit_hooks). For the cos(v_inf, theta_mc) cross-check we store the
    APPLIED direction (alpha*theta) keyed by (arr_idx, head) so the sign matches
    the trained shift; we also keep raw theta for reference.

    Returns (theta_mc_applied, theta_mc_raw, alpha_mc, info) where the first two
    are Tensor[n_captured_layers, num_q_heads, head_dim] (zeros where no mc head).
    A cell is filled only if its (layer, head) is an mc pair AND that layer is in
    captured_indices (same dominant-shape filter) AND theta head_dim matches.
    """
    layer_to_arr = {li: ai for ai, li in enumerate(captured_indices)}
    n_lay = len(captured_indices)
    theta_applied = torch.zeros(n_lay, num_q_heads, head_dim, dtype=torch.float32)
    theta_raw = torch.zeros(n_lay, num_q_heads, head_dim, dtype=torch.float32)
    alpha_grid = torch.zeros(n_lay, num_q_heads, dtype=torch.float32)
    info = {"path": MC_OFFSETS_PATH, "loaded": False, "n_pairs": 0,
            "n_aligned": 0, "head_dim_blob": None,
            "skipped_layer_not_captured": 0, "skipped_headdim_mismatch": 0}

    p = Path(MC_OFFSETS_PATH)
    if not p.exists():
        print(f"  WARNING: mc offsets not found at {MC_OFFSETS_PATH} — "
              f"theta_mc cross-check unavailable (zeros stored).", flush=True)
        return theta_applied, theta_raw, alpha_grid, info

    blob = torch.load(p, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in pr) for pr in blob["layer_head_pairs"]]
    alpha = blob["alpha"].to(torch.float32)
    theta = blob["theta"].to(torch.float32)
    blob_head_dim = int(blob.get("head_dim", theta.shape[-1]))
    info.update({"loaded": True, "n_pairs": len(pairs), "head_dim_blob": blob_head_dim})

    n_aligned = 0
    for i, (li, hi) in enumerate(pairs):
        if li not in layer_to_arr:
            info["skipped_layer_not_captured"] += 1
            continue
        if theta.shape[-1] != head_dim:
            info["skipped_headdim_mismatch"] += 1
            continue
        if hi < 0 or hi >= num_q_heads:
            continue
        ai = layer_to_arr[li]
        theta_raw[ai, hi] = theta[i]
        theta_applied[ai, hi] = alpha[i] * theta[i]
        alpha_grid[ai, hi] = alpha[i]
        n_aligned += 1
    info["n_aligned"] = n_aligned
    print(f"  mc offsets: {len(pairs)} pairs, aligned {n_aligned} to captured grid "
          f"(skipped: {info['skipped_layer_not_captured']} layer-not-captured, "
          f"{info['skipped_headdim_mismatch']} head_dim-mismatch).", flush=True)
    return theta_applied, theta_raw, alpha_grid, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="HF id or local path to Gemma 4 31B IT")
    ap.add_argument("--inference-datasets", default="arc,hellaswag",
                    help="Comma-separated DATASET_LOADERS keys (inference-form family)")
    ap.add_argument("--retrieval-datasets", default="phase1c_kr_sciq,phase1c_kr_medmcqa",
                    help="Comma-separated keys (retrieval-form family). Graceful "
                         "fallback to --retrieval-fallback if their JSONL is absent.")
    ap.add_argument("--retrieval-fallback", default="tqa",
                    help="Comma-separated keys used if retrieval JSONL absent "
                         "(default tqa = TruthfulQA mc1, HF dataset). Marked as "
                         "sensitivity-check, not primary.")
    ap.add_argument("--n-pairs", type=int, default=600,
                    help="Pairs PER dataset key; concatenated within each family.")
    ap.add_argument("--seq-len-max", type=int, default=384)
    ap.add_argument("--chat-template", action="store_true",
                    help="Wrap prompts in chat template (ON for IT models).")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="Seeds for per-head massmean CV accuracy.")
    ap.add_argument("--elig-acc", type=float, default=0.60,
                    help="Heads with massmean CV acc >= this in BOTH families are "
                         "'eligible'; full pooled cov is stored only for them.")
    ap.add_argument("--out", default="runs/oq1_functional_axis/functional_directions.pt")
    args = ap.parse_args()

    inf_keys = [x.strip() for x in args.inference_datasets.split(",") if x.strip()]
    ret_keys = [x.strip() for x in args.retrieval_datasets.split(",") if x.strip()]
    fb_keys = [x.strip() for x in args.retrieval_fallback.split(",") if x.strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OQ1-funcional (E2) — per-head functional axis extraction")
    print("=" * 60)
    print(f"  base: {args.base}")
    print(f"  inference family: {inf_keys}")
    print(f"  retrieval family: {ret_keys}  (fallback: {fb_keys})")
    print(f"  n_pairs/key: {args.n_pairs}   seq_len_max: {args.seq_len_max}")
    print(f"  chat_template: {args.chat_template}   elig_acc: {args.elig_acc}")
    print(f"  out: {out_path}")
    print()

    # -- Phase 1: build pairs for both families --------------------------------
    print("Phase 1: building contrastive pairs")
    inf_pairs, inf_used = _load_family_pairs(inf_keys, args.n_pairs)
    print(f"  inference family: {len(inf_pairs)} pairs from {inf_used}")
    ret_pairs, ret_used, ret_fallback = _load_retrieval_pairs(ret_keys, fb_keys, args.n_pairs)
    print(f"  retrieval family: {len(ret_pairs)} pairs from {ret_used} "
          f"(fallback_fired={ret_fallback})")
    print()

    # -- Phase 2: load model + detect attention shape (ONCE) -------------------
    print("Phase 2: loading model + runtime attention-shape detection")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    layers = _resolve_layers(model)
    n_layers = len(layers)
    attn_shape, layer_dims = _get_attention_shape_runtime(model, tokenizer, layers)
    num_q_heads = attn_shape.num_q_heads
    head_dim = attn_shape.head_dim
    print(f"  layers: {n_layers}")
    print(f"  num_q_heads: {num_q_heads}   head_dim: {head_dim}   "
          f"total_q_dim: {attn_shape.total_q_dim}")
    print(f"  hidden_size: {_resolve_text_config(model).hidden_size}")
    unique_dims = set(layer_dims.values())
    if len(unique_dims) > 1:
        print(f"  heterogeneous attention dims detected; non-dominant layers SKIPPED:")
        for li in sorted(layer_dims):
            if layer_dims[li] != attn_shape.total_q_dim:
                print(f"    L{li:2d}: dim={layer_dims[li]}")
    device = next(model.parameters()).device
    print()

    # -- Phase 3: capture per-head activations (4 forward sweeps) --------------
    # Same captured_indices for ALL sweeps (same model/attn_shape/layer_dims).
    print("Phase 3: capturing per-head activations (last token, o_proj pre-hook)")
    inf_correct_prompts, inf_wrong_prompts = _build_prompts(inf_pairs, args.chat_template, tokenizer)
    ret_correct_prompts, ret_wrong_prompts = _build_prompts(ret_pairs, args.chat_template, tokenizer)
    if args.chat_template:
        print(f"  chat-template ON — sample inf correct prompt[:200]:")
        print(f"    {inf_correct_prompts[0][:200]!r}")

    print(f"  [inf] capturing {len(inf_correct_prompts)} correct...", flush=True)
    H_inf_c, cap_inf = capture_per_head_activations(
        model, tokenizer, inf_correct_prompts, str(device),
        attn_shape, n_layers, args.seq_len_max, layer_dims,
    )
    print(f"  [inf] capturing {len(inf_wrong_prompts)} wrong...", flush=True)
    H_inf_w, _ = capture_per_head_activations(
        model, tokenizer, inf_wrong_prompts, str(device),
        attn_shape, n_layers, args.seq_len_max, layer_dims,
    )
    print(f"  [ret] capturing {len(ret_correct_prompts)} correct...", flush=True)
    H_ret_c, cap_ret = capture_per_head_activations(
        model, tokenizer, ret_correct_prompts, str(device),
        attn_shape, n_layers, args.seq_len_max, layer_dims,
    )
    print(f"  [ret] capturing {len(ret_wrong_prompts)} wrong...", flush=True)
    H_ret_w, _ = capture_per_head_activations(
        model, tokenizer, ret_wrong_prompts, str(device),
        attn_shape, n_layers, args.seq_len_max, layer_dims,
    )

    # Sanity: both families captured the SAME layer grid (single run guarantees
    # this; assert to fail loud rather than join on a mismatched grid).
    assert cap_inf == cap_ret, (
        f"captured_indices mismatch between families: inf={cap_inf} ret={cap_ret}"
    )
    captured_indices = cap_inf
    print(f"  H_inf_c {tuple(H_inf_c.shape)}  H_ret_c {tuple(H_ret_c.shape)}")
    print(f"  captured layer indices ({len(captured_indices)}): {captured_indices}")

    # Free GPU — everything below is CPU-bound (massmean / CV / cov).
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  GPU freed (del model + empty_cache); remaining compute is CPU-bound.")
    print()

    # -- Phase 4: cast to fp32 (CPU linalg lacks bf16 kernels; ~6-7x faster) ---
    print("Phase 4: massmean directions + per-class means + CV accuracy")
    if H_inf_c.dtype != torch.float32:
        print(f"  casting captures {H_inf_c.dtype} -> float32 for fast CPU linalg")
        H_inf_c = H_inf_c.to(torch.float32); H_inf_w = H_inf_w.to(torch.float32)
        H_ret_c = H_ret_c.to(torch.float32); H_ret_w = H_ret_w.to(torch.float32)

    print("  [inf] massmean directions + means")
    v_inf, mu_inf_c, mu_inf_w = _massmean_directions_and_means(
        H_inf_c, H_inf_w, captured_indices, num_q_heads)
    print("  [ret] massmean directions + means")
    v_ret, mu_ret_c, mu_ret_w = _massmean_directions_and_means(
        H_ret_c, H_ret_w, captured_indices, num_q_heads)

    print("  [inf] per-head massmean CV accuracy")
    acc_inf = _cv_acc_per_head(H_inf_c, H_inf_w, captured_indices, num_q_heads, args.n_seeds)
    print("  [ret] per-head massmean CV accuracy")
    acc_ret = _cv_acc_per_head(H_ret_c, H_ret_w, captured_indices, num_q_heads, args.n_seeds)
    print()

    # -- Phase 5: eligibility + pooled covariance for eligible cells -----------
    eligible_mask = (acc_inf >= args.elig_acc) & (acc_ret >= args.elig_acc)
    eligible_cells = [
        (int(ai), int(hi))
        for ai in range(len(captured_indices))
        for hi in range(num_q_heads)
        if bool(eligible_mask[ai, hi])
    ]
    n_elig = len(eligible_cells)
    print(f"Phase 5: eligibility + pooled covariance")
    print(f"  eligible heads (acc>={args.elig_acc} in BOTH families): "
          f"{n_elig}/{len(captured_indices) * num_q_heads}")
    if n_elig == 0:
        print("  WARNING: 0 eligible heads — storing NO covariance. The downstream "
              "probe can still use raw cos + per-class means, but whitened-cos on "
              "eligible heads will be empty. Consider lowering --elig-acc.", flush=True)
    # Store cov keyed by (layer, head) — the canonical join key used downstream.
    print("  [inf] pooled covariance for eligible cells")
    cov_inf_cells = _pooled_cov_for_cells(H_inf_c, H_inf_w, eligible_cells)
    print("  [ret] pooled covariance for eligible cells")
    cov_ret_cells = _pooled_cov_for_cells(H_ret_c, H_ret_w, eligible_cells)
    cov_inf = {(captured_indices[ai], hi): cov for (ai, hi), cov in cov_inf_cells.items()}
    cov_ret = {(captured_indices[ai], hi): cov for (ai, hi), cov in cov_ret_cells.items()}
    print()

    # -- Phase 6: mc-adapter theta alignment (cross-check cos(v_inf, theta_mc)) -
    print("Phase 6: aligning mc-adapter theta to captured grid")
    theta_mc_applied, theta_mc_raw, alpha_mc, mc_info = _load_theta_mc(
        captured_indices, num_q_heads, head_dim)
    print()

    # -- Save ------------------------------------------------------------------
    state = {
        "schema_version": SCHEMA_VERSION,
        # functional directions (the headline)
        "v_inf": v_inf,                       # [n_lay, num_q_heads, head_dim]
        "v_ret": v_ret,                       # [n_lay, num_q_heads, head_dim]
        # per-class means (cheap; lets probe recompute/sanity-check raw direction)
        "mu_inf_correct": mu_inf_c, "mu_inf_wrong": mu_inf_w,
        "mu_ret_correct": mu_ret_c, "mu_ret_wrong": mu_ret_w,
        # pooled within-class covariance for whitening — eligible heads only,
        # keyed by (layer, head) tuples (NOT arr_idx). See covariance_storage.
        "cov_inf": cov_inf, "cov_ret": cov_ret,
        # per-head probe accuracy in BOTH families (probe filters to acc>=0.60)
        "acc_inf": acc_inf, "acc_ret": acc_ret,
        "eligible_cells_layer_head": [
            [captured_indices[ai], hi] for (ai, hi) in eligible_cells
        ],
        "elig_acc": args.elig_acc,
        # mc-adapter cross-check: APPLIED direction (alpha*theta) is the trained
        # shift; raw theta + alpha kept for reference. Keyed on same grid.
        "theta_mc": theta_mc_applied,         # [n_lay, num_q_heads, head_dim] (alpha*theta)
        "theta_mc_raw": theta_mc_raw,         # [n_lay, num_q_heads, head_dim] (theta)
        "alpha_mc": alpha_mc,                 # [n_lay, num_q_heads]
        "mc_offsets_info": mc_info,
        # alignment metadata (the join key contract)
        "captured_indices": captured_indices,
        "num_q_heads": num_q_heads,
        "head_dim": head_dim,
        "total_q_dim": attn_shape.total_q_dim,
        "layer_dims": layer_dims,
        "n_layers_total": n_layers,
        # provenance
        "inference_datasets_requested": inf_keys,
        "inference_datasets_used": inf_used,
        "retrieval_datasets_requested": ret_keys,
        "retrieval_datasets_used": ret_used,
        "retrieval_fallback_fired": ret_fallback,
        "retrieval_fallback_keys": fb_keys,
        "n_pairs": args.n_pairs,
        "n_inf_pairs": len(inf_pairs),
        "n_ret_pairs": len(ret_pairs),
        "base": str(args.base),
        "chat_template": bool(args.chat_template),
        "seq_len_max": args.seq_len_max,
        "n_seeds": args.n_seeds,
        "covariance_storage": (
            "Full head_dim x head_dim pooled within-class covariance (fp32) stored "
            "per (layer,head) ONLY for ELIGIBLE heads (acc>=elig_acc in both families), "
            "separately per family (cov_inf, cov_ret). Centered on the POOLED mean "
            "exactly like park_whiten. Per-class means (mu_*) stored for ALL captured "
            "cells. Rationale: full cov for every cell is ~1GB/family at head_dim 256."
        ),
        "extraction_date": dt.datetime.now(dt.UTC).isoformat(),
    }
    torch.save(state, out_path)
    sz_mb = out_path.stat().st_size / (1024 * 1024)
    print("=" * 60)
    print(f"Saved: {out_path} ({sz_mb:.1f} MB)")
    print(f"  v_inf/v_ret grid: [{len(captured_indices)}, {num_q_heads}, {head_dim}]")
    print(f"  eligible heads (cov stored): {n_elig}")
    print(f"  mc theta aligned: {mc_info['n_aligned']} cells")
    print(f"  retrieval used: {ret_used} (fallback_fired={ret_fallback})")
    print("=" * 60)


if __name__ == "__main__":
    main()
