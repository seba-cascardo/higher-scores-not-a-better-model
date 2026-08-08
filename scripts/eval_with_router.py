"""V8 router-aware Tier-1 eval — gate V7 offsets per-prompt via classifier.

Setup:
  - Load BASE Gemma 4 E2B (no V7 modifications).
  - Load V7 trained offsets (runs/v7_lofit/offsets_clean.pt).
  - Load trained router (runs/v8_router/router.pt).
  - For each Tier-1 prompt:
      1. Router classifies prompt → "mc" or "non-mc"
      2. If "mc": install LoFiT pre-hooks with V7 offsets, score prompt, remove
      3. If "non-mc": score prompt with base model, no hooks

Compares against:
  - V7 always-on (eval_tier1_v7.json existing): all prompts get offsets
  - Base always-off (no eval yet): no prompts get offsets
  Goal: V7-with-router should approximately match V7 on mc tasks AND base on
  non-mc tasks. Pareto-optimal IF and ONLY IF router classifies accurately.

Routing variants:
  --router-variant: text_formatted | text_stripped | activation
    text_formatted: TF-IDF on prompt-as-fed-to-model (with "Question:..." prefix)
    text_stripped : TF-IDF on stripped prompt (no formatting)
    activation    : layer-K residual stream mean-pool, trained classifier

Usage:
    python -m scripts.eval_with_router \\
        --base /path/to/models/gemma4-e2b \\
        --offsets runs/v7_lofit/offsets_clean.pt \\
        --router runs/v8_router/router.pt \\
        --router-variant activation \\
        --tasks lambada_openai,truthfulqa,arc_challenge,hellaswag,winogrande,gsm8k \\
        --limit 400 \\
        --out runs/v8_router/eval_with_router.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -- Layer / config resolution ------------------------------------------------
def _resolve_layers(model):
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
    cfg = model.config
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


@dataclass
class AttentionShape:
    num_q_heads: int
    head_dim: int


def _get_attention_shape(model, layers=None) -> AttentionShape:
    cfg = _resolve_text_config(model)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        num_h = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = hidden // num_h
    if layers is None:
        layers = _resolve_layers(model)
    # Hybrid-attention models (Qwen 3.5/3.6): layer 0 may be linear-attention
    # (no .self_attn). Find the first layer with self_attn.
    full_attn_layer = next(
        (l for l in layers if hasattr(l, "self_attn")), None,
    )
    if full_attn_layer is None:
        raise AssertionError("No layer with self_attn found — model may not support LoFiT")
    o_proj = full_attn_layer.self_attn.o_proj
    total_q_dim = o_proj.in_features
    if total_q_dim % head_dim != 0:
        raise AssertionError(
            f"o_proj.in_features={total_q_dim} not divisible by head_dim={head_dim}"
        )
    num_q_heads = total_q_dim // head_dim
    return AttentionShape(num_q_heads=num_q_heads, head_dim=head_dim)


# -- LoFiT hook installation (mirrors train_v7_lofit.py) ---------------------
def install_lofit_hooks(layers, alpha, theta, layer_head_pairs, attn_shape):
    """Install pre-hooks adding precomputed α·θ to per-head slice of o_proj input.

    Vectorized inference variant: alpha and theta are FIXED at inference time
    (no grad), so we precompute a flat (total,) layer-offset tensor once at
    install time on o_proj's device + dtype. The hot path becomes a single
    broadcast-add per token per layer — one CUDA kernel, zero allocations.

    This eliminates the per-token Python loop and per-token (B, T, num_q_heads,
    head_dim) allocation that the training-time hook needs (training requires
    grad through alpha/theta; inference does not).

    Numerical identity vs the prior implementation: for selected (li, hi)
    pairs the per-coordinate value `modified[b, t, hi*head_dim + d] =
    concat[b, t, hi*head_dim + d] + (α_h · θ_h)_d` is identical, since both
    versions cast α and θ to o_proj's bf16 dtype before multiplying and adding.

    alpha: (n_pairs,) — fp32 or bf16
    theta: (n_pairs, head_dim) — fp32 or bf16
    layer_head_pairs: list[(layer, head)]
    """
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for idx, (li, hi) in enumerate(layer_head_pairs):
        by_layer.setdefault(li, []).append((hi, idx))

    expected_total = attn_shape.num_q_heads * attn_shape.head_dim
    num_q_heads = attn_shape.num_q_heads
    head_dim = attn_shape.head_dim
    handles = []
    for layer_idx, items in by_layer.items():
        layer = layers[layer_idx]
        # Defensive — for hybrid-attention models, the offsets blob should only
        # contain (full-attention layer, head) pairs. Skip if we somehow got a
        # linear-attention layer index.
        if not hasattr(layer, "self_attn"):
            print(f"  WARN: layer {layer_idx} has no self_attn (linear-attention?), skipping")
            continue
        o_proj = layer.self_attn.o_proj

        # Precompute flat (total,) layer offset on o_proj's device + dtype.
        # This happens ONCE at install time; the per-token hook just broadcasts.
        target_device = o_proj.weight.device
        target_dtype = o_proj.weight.dtype
        layer_offset = torch.zeros(
            num_q_heads, head_dim, dtype=target_dtype, device=target_device
        )
        for hi, idx in items:
            a = alpha[idx].to(device=target_device, dtype=target_dtype)
            th = theta[idx].to(device=target_device, dtype=target_dtype)
            layer_offset[hi] = a * th
        layer_offset_flat = layer_offset.view(num_q_heads * head_dim)  # (total,)

        def make_hook(li, offset_flat):
            def hook(_module, inputs):
                concat = inputs[0]
                B, T, total = concat.shape
                if total != expected_total:
                    raise RuntimeError(
                        f"L{li}: o_proj input dim {total} != expected {expected_total}"
                    )
                # Single CUDA kernel: broadcast (total,) over (B, T, total) and add.
                return (concat + offset_flat,) + inputs[1:]
            return hook

        handles.append(
            o_proj.register_forward_pre_hook(make_hook(layer_idx, layer_offset_flat))
        )
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()


# -- Router wrapper ----------------------------------------------------------
class Router:
    """Predicts mc(1) vs non-mc(0) for a prompt, using one of three variants."""

    def __init__(self, blob: dict, variant: str, model=None, tokenizer=None):
        self.variant = variant
        self.target_layer = blob["config"].get("target_layer", 5)
        if variant == "text_formatted":
            self.vec = pickle.loads(blob["vec_formatted_pickle"])
            self.clf = pickle.loads(blob["clf_formatted_pickle"])
        elif variant == "text_stripped":
            self.vec = pickle.loads(blob["vec_stripped_pickle"])
            self.clf = pickle.loads(blob["clf_stripped_pickle"])
        elif variant == "activation":
            self.clf = pickle.loads(blob["clf_activation_pickle"])
            if model is None or tokenizer is None:
                raise ValueError("activation variant needs model + tokenizer")
            self.model = model
            self.tokenizer = tokenizer
        else:
            raise ValueError(f"Unknown variant {variant}")

    @torch.no_grad()
    def predict(self, prompt: str) -> tuple[int, float]:
        """Returns (label, mc_probability).

        label: 1 = mc, 0 = non-mc
        """
        if self.variant in ("text_formatted", "text_stripped"):
            X = self.vec.transform([prompt])
            proba = float(self.clf.predict_proba(X)[0, 1])
            label = int(proba >= 0.5)
            return label, proba
        elif self.variant == "activation":
            # Forward through model, capture target layer activation
            layers = _resolve_layers(self.model)
            captured = {}

            def hook(_m, _i, output):
                captured["x"] = output[0] if isinstance(output, tuple) else output

            h = layers[self.target_layer - 1].register_forward_hook(hook)
            try:
                toks = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                ).to(self.model.device)
                self.model(**toks, use_cache=False)
            finally:
                h.remove()
            pooled = captured["x"][0].mean(dim=0).to(dtype=torch.float32, device="cpu").numpy()
            proba = float(self.clf.predict_proba(pooled.reshape(1, -1))[0, 1])
            label = int(proba >= 0.5)
            return label, proba


# -- Eval routines (with router gating) --------------------------------------
class RouterAwareScorer:
    """Manages hook install/remove around scoring functions based on router decision."""

    def __init__(self, model, tokenizer, layers, attn_shape,
                 alpha, theta, layer_head_pairs, router):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.attn_shape = attn_shape
        self.alpha = alpha
        self.theta = theta
        self.layer_head_pairs = layer_head_pairs
        self.router = router
        self.routing_log = []  # list of (task, prompt_preview, label, proba)

    def maybe_install(self, prompt: str, task_name: str = "") -> tuple[list, int, float]:
        label, proba = self.router.predict(prompt)
        self.routing_log.append({
            "task": task_name,
            "prompt_preview": prompt[:80].replace("\n", " "),
            "label": label,
            "mc_proba": proba,
        })
        if label == 1:
            handles = install_lofit_hooks(
                self.layers, self.alpha, self.theta,
                self.layer_head_pairs, self.attn_shape,
            )
        else:
            handles = []
        return handles, label, proba


@torch.no_grad()
def loglikelihood(model, tok, prompt: str, continuation: str) -> tuple[float, int]:
    full = prompt + continuation
    full_ids = tok(full, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    n_prompt = prompt_ids.shape[1]
    n_continuation = full_ids.shape[1] - n_prompt
    if n_continuation <= 0:
        return 0.0, 0
    out = model(full_ids)
    logits = out.logits[0, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = full_ids[0, 1:]
    cont_log_probs = log_probs[n_prompt - 1:, :].gather(
        -1, targets[n_prompt - 1:].unsqueeze(-1)
    ).squeeze(-1)
    return float(cont_log_probs.sum().item()), int(cont_log_probs.numel())


def _argmax_mc_routed(scorer, prompt: str, choices: list[str], task_name: str) -> tuple[int, int, float]:
    """MC argmax with router gating. Hook installed once per question, not per choice."""
    handles, label, proba = scorer.maybe_install(prompt, task_name)
    try:
        lls = []
        for c in choices:
            lp, n = loglikelihood(scorer.model, scorer.tokenizer, prompt, c)
            lls.append(lp / max(n, 1))
        return int(max(range(len(lls)), key=lambda i: lls[i])), label, proba
    finally:
        remove_handles(handles)


# -- Per-task evals ----------------------------------------------------------
def eval_lambada(scorer, max_samples: int = 400) -> dict:
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    n = min(max_samples, len(ds))
    n_correct = 0
    n_total = 0
    sum_lp = 0.0
    sum_len = 0
    routing_stats = {"mc": 0, "non_mc": 0}
    for row in ds.select(range(n)):
        text = row["text"].strip()
        if not text:
            continue
        words = text.split()
        if len(words) < 2:
            continue
        prompt = " ".join(words[:-1])
        target = " " + words[-1]

        handles, label, proba = scorer.maybe_install(prompt, "lambada")
        routing_stats["mc" if label else "non_mc"] += 1
        try:
            full_ids = scorer.tokenizer(prompt + target, return_tensors="pt",
                                        add_special_tokens=True).input_ids.to(scorer.model.device)
            prompt_ids = scorer.tokenizer(prompt, return_tensors="pt",
                                          add_special_tokens=True).input_ids.to(scorer.model.device)
            n_prompt = prompt_ids.shape[1]
            with torch.no_grad():
                out = scorer.model(full_ids)
            logits = out.logits[0, :-1].float()
            log_probs = F.log_softmax(logits, dim=-1)
            targets = full_ids[0, 1:]
            cont_lp = log_probs[n_prompt - 1:].gather(-1, targets[n_prompt - 1:].unsqueeze(-1)).squeeze(-1)
            target_argmax = logits[n_prompt - 1:].argmax(dim=-1)
            correct = bool(torch.equal(target_argmax, targets[n_prompt - 1:]))
            n_correct += int(correct)
            n_total += 1
            sum_lp += float(cont_lp.sum().item())
            sum_len += cont_lp.numel()
        finally:
            remove_handles(handles)

    acc = n_correct / max(n_total, 1)
    ppl = math.exp(-sum_lp / max(sum_len, 1)) if sum_len else float("inf")
    return {"accuracy": acc, "perplexity": ppl, "n": n_total, "routing": routing_stats}


def eval_truthfulqa(scorer, max_samples: int = 400) -> dict:
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    n = min(max_samples, len(ds))
    correct_mc1 = 0
    sum_mc2 = 0.0
    n_total = 0
    routing_stats = {"mc": 0, "non_mc": 0}
    for row in ds.select(range(n)):
        q = "Q: " + row["question"].strip() + "\nA:"
        mc1_choices = [" " + c.strip() for c in row["mc1_targets"]["choices"]]
        mc1_labels = row["mc1_targets"]["labels"]
        mc1_pred, label, proba = _argmax_mc_routed(scorer, q, mc1_choices, "truthfulqa")
        routing_stats["mc" if label else "non_mc"] += 1
        correct_mc1 += int(mc1_labels[mc1_pred] == 1)
        # mc2 — use SAME routing decision (cheaper than re-classifying)
        handles = []
        if label == 1:
            handles = install_lofit_hooks(
                scorer.layers, scorer.alpha, scorer.theta,
                scorer.layer_head_pairs, scorer.attn_shape,
            )
        try:
            mc2_choices = [" " + c.strip() for c in row["mc2_targets"]["choices"]]
            mc2_labels = row["mc2_targets"]["labels"]
            per_choice_lp = []
            for c in mc2_choices:
                lp, _ = loglikelihood(scorer.model, scorer.tokenizer, q, c)
                per_choice_lp.append(lp)
            lps = torch.tensor(per_choice_lp)
            norm = torch.logsumexp(lps, dim=0)
            tmask = torch.tensor([bool(l) for l in mc2_labels])
            if tmask.any():
                tnorm = torch.logsumexp(lps[tmask], dim=0)
                mc2_score = float(torch.exp(tnorm - norm).item())
            else:
                mc2_score = 0.0
            sum_mc2 += mc2_score
            n_total += 1
        finally:
            remove_handles(handles)
    return {
        "mc1_accuracy": correct_mc1 / max(n_total, 1),
        "mc2_accuracy": sum_mc2 / max(n_total, 1),
        "n": n_total, "routing": routing_stats,
    }


def eval_hellaswag(scorer, max_samples: int = 400) -> dict:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    n = min(max_samples, len(ds))
    n_correct = 0
    n_total = 0
    routing_stats = {"mc": 0, "non_mc": 0}
    for row in ds.select(range(n)):
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        endings = [" " + e for e in row["endings"]]
        gold = int(row["label"])
        pred, label, _ = _argmax_mc_routed(scorer, ctx, endings, "hellaswag")
        routing_stats["mc" if label else "non_mc"] += 1
        n_correct += int(pred == gold)
        n_total += 1
    return {"acc_norm": n_correct / max(n_total, 1), "n": n_total, "routing": routing_stats}


def eval_arc(scorer, max_samples: int = 400) -> dict:
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    except Exception:
        ds = load_dataset(
            "parquet",
            data_files={"test": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet"},
            split="test",
        )
    n = min(max_samples, len(ds))
    n_correct = 0
    n_total = 0
    routing_stats = {"mc": 0, "non_mc": 0}
    for row in ds.select(range(n)):
        q = "Question: " + row["question"].strip() + "\nAnswer:"
        choices_text = row["choices"]["text"]
        labels = row["choices"]["label"]
        choices = [" " + c.strip() for c in choices_text]
        gold = labels.index(row["answerKey"]) if row["answerKey"] in labels else 0
        pred, label, _ = _argmax_mc_routed(scorer, q, choices, "arc_challenge")
        routing_stats["mc" if label else "non_mc"] += 1
        n_correct += int(pred == gold)
        n_total += 1
    return {"acc_norm": n_correct / max(n_total, 1), "n": n_total, "routing": routing_stats}


def eval_winogrande(scorer, max_samples: int = 400) -> dict:
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    n = min(max_samples, len(ds))
    n_correct = 0
    n_total = 0
    routing_stats = {"mc": 0, "non_mc": 0}
    for row in ds.select(range(n)):
        sentence = row["sentence"]
        # Route based on the sentence (winogrande has no Q/A format)
        handles, label, _ = scorer.maybe_install(sentence, "winogrande")
        routing_stats["mc" if label else "non_mc"] += 1
        try:
            s1 = sentence.replace("_", row["option1"].strip())
            s2 = sentence.replace("_", row["option2"].strip())
            lp1, n1 = loglikelihood(scorer.model, scorer.tokenizer, "", s1)
            lp2, n2 = loglikelihood(scorer.model, scorer.tokenizer, "", s2)
            avg1 = lp1 / max(n1, 1)
            avg2 = lp2 / max(n2, 1)
            pred_str = "1" if avg1 > avg2 else "2"
            n_correct += int(pred_str == row["answer"])
            n_total += 1
        finally:
            remove_handles(handles)
    return {"accuracy": n_correct / max(n_total, 1), "n": n_total, "routing": routing_stats}


@torch.no_grad()
def eval_gsm8k(scorer, max_samples: int = 200, max_new_tokens: int = 128) -> dict:
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(max_samples, len(ds))
    n_correct = 0
    n_total = 0
    routing_stats = {"mc": 0, "non_mc": 0}
    for row in ds.select(range(n)):
        prompt = "Question: " + row["question"].strip() + "\nAnswer:"
        handles, label, _ = scorer.maybe_install(prompt, "gsm8k")
        routing_stats["mc" if label else "non_mc"] += 1
        try:
            ids = scorer.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(scorer.model.device)
            out = scorer.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=scorer.tokenizer.eos_token_id,
            )
            gen = scorer.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            nums = re.findall(r"-?\d+(?:\.\d+)?", gen.replace(",", ""))
            pred = nums[-1] if nums else None
            gold_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", row["answer"])
            gold = gold_match.group(1) if gold_match else None
            if pred is not None and gold is not None:
                try:
                    if abs(float(pred) - float(gold)) < 1e-6:
                        n_correct += 1
                except ValueError:
                    pass
            n_total += 1
        finally:
            remove_handles(handles)
    return {"accuracy": n_correct / max(n_total, 1), "n": n_total, "routing": routing_stats}


# -- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--offsets", required=True,
                    help="V7 offsets file (e.g. runs/v7_lofit/offsets_clean.pt)")
    ap.add_argument("--router", required=True,
                    help="Router .pt file from train_router_v8.py")
    ap.add_argument("--router-variant", default="activation",
                    choices=["text_formatted", "text_stripped", "activation"])
    ap.add_argument("--tasks", default="lambada_openai,truthfulqa,arc_challenge,hellaswag,winogrande,gsm8k")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--gsm8k-limit", type=int, default=200,
                    help="GSM8K is slow (generation); use smaller limit by default")
    ap.add_argument("--gsm8k-max-new-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V8 Router-Aware Tier-1 Eval")
    print("=" * 60)
    print(f"  base: {args.base}")
    print(f"  offsets: {args.offsets}")
    print(f"  router: {args.router} (variant={args.router_variant})")
    print(f"  tasks: {args.tasks}")
    print(f"  limit: {args.limit}, gsm8k_limit: {args.gsm8k_limit}")
    print()

    # Load model
    print("Phase 1: load base model")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    layers = _resolve_layers(model)
    attn_shape = _get_attention_shape(model, layers)
    print(f"  layers={len(layers)} num_q_heads={attn_shape.num_q_heads} head_dim={attn_shape.head_dim}")

    # Load V7 offsets
    print("Phase 2: load V7 offsets")
    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    layer_head_pairs = blob["layer_head_pairs"]
    alpha = blob["alpha"]
    theta = blob["theta"]
    print(f"  loaded {len(layer_head_pairs)} (layer, head) pairs")
    print(f"  alpha shape: {alpha.shape}, theta shape: {theta.shape}")

    # Filter to compatible layers (sliding-attention)
    expected_total = attn_shape.num_q_heads * attn_shape.head_dim
    # Use dummy forward to detect dims
    detected = {}
    handles = []
    for li, layer in enumerate(layers):
        op = layer.self_attn.o_proj
        def make_hook(idx):
            def hook(_m, ins): detected[idx] = ins[0].shape[-1]
            return hook
        handles.append(op.register_forward_pre_hook(make_hook(li)))
    toks = tokenizer("Hi", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**toks, use_cache=False)
    for h in handles: h.remove()

    compatible_layers = {li for li, d in detected.items() if d == expected_total}
    filtered_pairs = []
    keep_idx = []
    for idx, (li, hi) in enumerate(layer_head_pairs):
        if li in compatible_layers:
            filtered_pairs.append((li, hi))
            keep_idx.append(idx)
    if len(filtered_pairs) < len(layer_head_pairs):
        print(f"  filtered {len(layer_head_pairs)} -> {len(filtered_pairs)} compatible heads")
        alpha = alpha[keep_idx]
        theta = theta[keep_idx]
        layer_head_pairs = filtered_pairs

    # Load router
    print(f"Phase 3: load router (variant={args.router_variant})")
    router_blob = torch.load(args.router, map_location="cpu", weights_only=False)
    router = Router(router_blob, args.router_variant, model=model, tokenizer=tokenizer)
    print(f"  router.target_layer={router.target_layer if args.router_variant == 'activation' else 'n/a'}")
    print(f"  trained accuracy on test: {router_blob['results'].get(args.router_variant, {}).get('accuracy', 'n/a')}")
    print()

    scorer = RouterAwareScorer(
        model=model, tokenizer=tokenizer, layers=layers, attn_shape=attn_shape,
        alpha=alpha, theta=theta, layer_head_pairs=layer_head_pairs, router=router,
    )

    # Run tasks
    print("Phase 4: run Tier-1 with routing")
    print()
    results = {"config": vars(args), "tasks": {}}
    selected = [t.strip() for t in args.tasks.split(",")]
    for task in selected:
        print(f"  → {task} (limit={args.gsm8k_limit if task == 'gsm8k' else args.limit})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.time()
        try:
            if task == "lambada_openai":
                r = eval_lambada(scorer, args.limit)
            elif task == "truthfulqa":
                r = eval_truthfulqa(scorer, args.limit)
            elif task == "arc_challenge":
                r = eval_arc(scorer, args.limit)
            elif task == "hellaswag":
                r = eval_hellaswag(scorer, args.limit)
            elif task == "winogrande":
                r = eval_winogrande(scorer, args.limit)
            elif task == "gsm8k":
                r = eval_gsm8k(scorer, args.gsm8k_limit, args.gsm8k_max_new_tokens)
            else:
                r = {"error": f"unknown task {task}"}
            elapsed = time.time() - t0
            r["elapsed_s"] = elapsed
            results["tasks"][task] = r
            print(f"    {r}  [{elapsed:.0f}s]")
        except Exception as e:
            results["tasks"][task] = {"error": str(e)}
            print(f"    FAILED: {e}")
        # Periodic save
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # Save final
    results["routing_log"] = scorer.routing_log[:500]  # cap for size
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print()
    print(f"Saved: {out_path}")

    # Quick summary
    print()
    print("=" * 60)
    print("Per-task routing breakdown")
    print("=" * 60)
    for task, r in results["tasks"].items():
        if "routing" in r:
            mc_n = r["routing"].get("mc", 0)
            nonmc_n = r["routing"].get("non_mc", 0)
            total = mc_n + nonmc_n
            mc_pct = 100 * mc_n / max(total, 1)
            print(f"  {task:18s}  mc={mc_n}/{total} ({mc_pct:5.1f}%)  non_mc={nonmc_n}")


if __name__ == "__main__":
    main()
