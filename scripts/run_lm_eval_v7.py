"""V7 LoFiT × lm-evaluation-harness canonical eval (P1 of 2026-05-09 plan).

Provides a custom HFLM subclass `V7HFLM` that loads a base model + V7 LoFiT
offsets and installs hooks at construction, then delegates to lm-evaluation-
harness's standard scoring path. Result: fully apples-to-apples deltas vs
canonical `lm_eval` baselines (char-norm, raw chat-template wrap, no custom
instruction prefixes).

Closes the methodology gap documented in
`memory/feedback_paper_findings_2026_05_08.md` Finding 3 (Caveat 8).

## Usage

Canonical eval (P1):
    python -m scripts.run_lm_eval_v7 \\
        --base /workspace/models/gemma4-31b-it \\
        --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \\
        --tasks arc_challenge,truthfulqa_mc1,hellaswag,winogrande \\
        --limit 400 \\
        --apply-chat-template \\
        --out runs/v7_lofit_gemma4_31b_chat/eval/eval_mc_lm_eval_canonical.json

LM-preservation smoke (P2.a — V7-completion safety gate):
    python -m scripts.run_lm_eval_v7 \\
        --base /workspace/models/gemma4-31b-it \\
        --offsets runs/v7_lofit_gemma4_31b_chat/offsets_completion.pt \\
        --ppl-smoke \\
        --out runs/v7_lofit_gemma4_31b_chat/eval/ppl_smoke_completion.json

## Train/eval contamination notes (read together with paper §5 Limitations)

V7-mc adapter trained on (TQA + ARC) per-task seed-0 perm:
  - ARC:   train split (HF native)              vs lm-eval test split          → DISJOINT BY CONSTRUCTION ✓
  - TQA:   first 80% of validation (seed-0)     vs lm-eval validation order     → PARTIAL OVERLAP — caveat
  - HSwag: NOT in V7-mc training                vs lm-eval validation           → CLEAN TRANSFER ✓
  - Wino:  NOT in V7-mc training                vs lm-eval validation           → CLEAN TRANSFER ✓

V7-completion adapter trained on Lambada seed-0 perm (test split, first 80%).
  Lambada chat-template eval is broken in lm-eval (Finding 5) → skip Lambada
  for V7-completion canonical eval. Use ARC/HSwag/TQA/Wino as transfer probes.

For paper claim wording:
  - "ARC challenge +Xpp (clean, train/test disjoint)"
  - "HellaSwag +Xpp (clean transfer, no HSwag in V7 training)"
  - "Winogrande +Xpp (clean transfer)"
  - "TruthfulQA mc1 +Xpp (partial validation overlap with V7-mc training, see appendix)"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Multimodal fallback for Gemma 4 31B IT / Qwen 3.6 27B which register as
# *ForImageTextToText in transformers ≥5.0
try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

# Reuse hook installation + layer resolution from existing infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_with_router import (  # noqa: E402
    install_lofit_hooks,
    remove_handles,
    _resolve_layers,
    _get_attention_shape,
)


# -- Model loading with multimodal fallback ----------------------------------
def _load_model_multimodal(base_path: str, dtype: torch.dtype):
    """Load base model, falling back to ImageTextToText for multimodal-registered repos."""
    last_err = None
    if AutoModelForCausalLM is not None:
        try:
            return AutoModelForCausalLM.from_pretrained(
                base_path, dtype=dtype, device_map="cuda",
            )
        except (ValueError, KeyError) as e:
            last_err = e
            print(f"[v7hf] AutoModelForCausalLM failed ({e}), trying ImageTextToText...")
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText.from_pretrained(
            base_path, dtype=dtype, device_map="cuda",
        )
    raise RuntimeError(
        f"Could not load {base_path} as Causal LM or ImageTextToText. Last error: {last_err}"
    )


def _filter_compatible_layers(model, tokenizer, attn_shape, layer_head_pairs, alpha, theta):
    """Drop layer_head_pairs whose layer's o_proj input dim != dominant.

    Mirrors logic from train_v7_lofit.py and eval_with_router.py main(). For
    Gemma 4: filters out the alternating sliding-attention layers with smaller
    o_proj input dim. For Qwen 3.5/3.6 hybrid attention: filters out
    linear-attention layers without self_attn.
    """
    layers = _resolve_layers(model)
    expected_total = attn_shape.num_q_heads * attn_shape.head_dim
    detected: dict[int, int] = {}
    handles = []
    for li, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            continue
        op = layer.self_attn.o_proj

        def make_hook(idx):
            def hook(_m, ins):
                detected[idx] = ins[0].shape[-1]
            return hook
        handles.append(op.register_forward_pre_hook(make_hook(li)))
    toks = tokenizer("Hi", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**toks, use_cache=False)
    for h in handles:
        h.remove()

    compatible_layers = {li for li, d in detected.items() if d == expected_total}
    filtered_pairs = []
    keep_idx = []
    for idx, (li, hi) in enumerate(layer_head_pairs):
        if li in compatible_layers:
            filtered_pairs.append((li, hi))
            keep_idx.append(idx)
    if len(filtered_pairs) < len(layer_head_pairs):
        print(f"[v7hf] filtered {len(layer_head_pairs)} -> {len(filtered_pairs)} compatible (layer, head) pairs")
        alpha = alpha[keep_idx]
        theta = theta[keep_idx]
    return filtered_pairs, alpha, theta


# -- Custom HFLM subclass for lm-evaluation-harness ---------------------------
def _register_v7hflm():
    """Register V7HFLM lazily so this script can be imported without lm_eval installed
    (e.g. for --ppl-smoke mode on a minimal env). Call from main() before delegating
    to lm_eval.simple_evaluate.
    """
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM

    @register_model("v7hf")
    class V7HFLM(HFLM):
        """HFLM subclass that loads V7 LoFiT offsets and installs hooks at construction.

        Constructor accepts the standard model_args string from lm_eval CLI:
          model_args="base=/path/to/model,offsets=/path/to/offsets.pt,dtype=bfloat16"

        After loading the model + tokenizer, V7 hooks are installed and remain
        active for the lifetime of the V7HFLM instance. lm_eval's standard
        loglikelihood / generate path then runs through the V7-modified model.
        """

        def __init__(
            self,
            base: str,
            offsets: str,
            dtype: str = "bfloat16",
            null_adapter: bool = False,
            layer_lo=None,
            layer_hi=None,
            offset_scale=1.0,
            **kwargs,
        ):
            torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            print(f"[v7hf] loading base model from {base} (dtype={dtype})...")
            t0 = time.time()
            model = _load_model_multimodal(base, torch_dtype)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(base)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"[v7hf] base loaded in {time.time()-t0:.1f}s")

            super().__init__(pretrained=model, tokenizer=tokenizer, **kwargs)

            print(f"[v7hf] loading V7 offsets from {offsets}...")
            blob = torch.load(offsets, map_location="cpu", weights_only=False)
            layer_head_pairs = blob["layer_head_pairs"]
            alpha = blob["alpha"]
            theta = blob["theta"]
            print(f"[v7hf]   {len(layer_head_pairs)} (layer, head) pairs; alpha {alpha.shape}, theta {theta.shape}")

            # Causal layer-band patching (localization control): keep only adapter heads
            # whose layer is in [layer_lo, layer_hi]. None on both = no filter (canonical
            # d_mc reproduced exactly). Filters layer_head_pairs/alpha/theta consistently
            # BEFORE _filter_compatible_layers + hook install.
            if layer_lo is not None or layer_hi is not None:
                lo = -1 if layer_lo is None else int(layer_lo)
                hi = 10**9 if layer_hi is None else int(layer_hi)
                n0 = len(layer_head_pairs)
                keep = [j for j in range(n0) if lo <= int(layer_head_pairs[j][0]) <= hi]
                layer_head_pairs = [layer_head_pairs[j] for j in keep]
                alpha = alpha[keep]
                theta = theta[keep]
                kept_layers = sorted({int(p[0]) for p in layer_head_pairs})
                print(f"[v7hf]   LAYER-BAND [{lo},{hi}]: kept {len(keep)}/{n0} heads; "
                      f"layers={kept_layers}")

            if null_adapter:
                alpha = torch.zeros_like(alpha)
                theta = torch.zeros_like(theta)
                print("[v7hf]   NULL-ADAPTER mode: alpha/theta zeroed -> offsets add 0; "
                      "the v7hf wrapper (load + filter + hook install + hook exec) runs "
                      "identically but is a behavioral identity. Pre-reg R1/R7.")

            layers = _resolve_layers(model)
            attn_shape = _get_attention_shape(model, layers)
            layer_head_pairs, alpha, theta = _filter_compatible_layers(
                model, tokenizer, attn_shape, layer_head_pairs, alpha, theta,
            )
            # Activation-scale strength sweep (P-dose): multiply the trained
            # offset magnitude by offset_scale. 1.0 = canonical (+35 headline);
            # 0.0 = exact base (alpha*0 reproduces the zeroed-offsets base arm,
            # so --offset-scale 0 == base is the first sanity to run). Same axis
            # as score_mc_premise_pod.py:228, so this acc_norm dose-curve is
            # directly comparable to the DC-PMI + gsm8k activation-scale curves.
            scaled_alpha = alpha * float(offset_scale)
            if abs(float(offset_scale) - 1.0) > 1e-9:
                print(f"[v7hf]   OFFSET-SCALE {offset_scale}: alpha magnitude scaled "
                      f"(1.0=canonical, 0.0=base)")
            self._v7_handles = install_lofit_hooks(
                layers, scaled_alpha, theta, layer_head_pairs, attn_shape,
            )
            unique_layers = sorted({li for li, _ in layer_head_pairs})
            print(f"[v7hf] installed {len(self._v7_handles)} pre-hooks across {len(unique_layers)} unique layers")
            print(f"[v7hf] modulated layers: {unique_layers}")

        def __del__(self):
            handles = getattr(self, "_v7_handles", None)
            if handles:
                remove_handles(handles)

    return V7HFLM


# -- LM-preservation smoke (no lm_eval dependency) ---------------------------
SMOKE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "The capital of France is Paris, a city known for its art and history.",
]


@torch.no_grad()
def _ppl_on_sentences(model, tokenizer, sentences: list[str]) -> dict:
    """Compute mean negative log-likelihood per token, then exp() for PPL.

    Uses standard causal LM scoring: shift logits, gather token log-probs.
    """
    total_lp = 0.0
    total_tok = 0
    per_sent = []
    for s in sentences:
        ids = tokenizer(s, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        out = model(ids)
        logits = out.logits[0, :-1]
        targets = ids[0, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        sum_lp = float(token_lps.sum().item())
        n_tok = int(token_lps.numel())
        total_lp += sum_lp
        total_tok += n_tok
        per_sent.append({
            "text": s,
            "n_tokens": n_tok,
            "mean_nll": -sum_lp / max(n_tok, 1),
            "ppl": float(torch.exp(torch.tensor(-sum_lp / max(n_tok, 1))).item()),
        })
    mean_nll = -total_lp / max(total_tok, 1)
    return {
        "mean_nll": mean_nll,
        "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
        "n_sentences": len(per_sent),
        "n_tokens_total": total_tok,
        "per_sentence": per_sent,
    }


def run_ppl_smoke(args):
    """Run V7 vs base PPL smoke on 5 English sentences. Decision gate per handoff:
    if V7 PPL ratio > 5× base → V7-reasoning failure mode replicated, abort Tier-1 plan.
    """
    torch_dtype = getattr(torch, args.dtype)
    print(f"[smoke] loading base from {args.base}...")
    model = _load_model_multimodal(args.base, torch_dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"[smoke] loading V7 offsets from {args.offsets}...")
    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    layer_head_pairs = blob["layer_head_pairs"]
    alpha = blob["alpha"]
    theta = blob["theta"]
    layers = _resolve_layers(model)
    attn_shape = _get_attention_shape(model, layers)
    layer_head_pairs, alpha, theta = _filter_compatible_layers(
        model, tokenizer, attn_shape, layer_head_pairs, alpha, theta,
    )

    print("\n[smoke] === BASE (no V7) ===")
    base_result = _ppl_on_sentences(model, tokenizer, SMOKE_SENTENCES)
    print(f"  mean PPL = {base_result['ppl']:.3f}  (n_tok={base_result['n_tokens_total']})")

    print("\n[smoke] === V7 INSTALLED ===")
    handles = install_lofit_hooks(
        layers, alpha * float(args.offset_scale), theta, layer_head_pairs, attn_shape,
    )
    try:
        v7_result = _ppl_on_sentences(model, tokenizer, SMOKE_SENTENCES)
    finally:
        remove_handles(handles)
    print(f"  mean PPL = {v7_result['ppl']:.3f}  (n_tok={v7_result['n_tokens_total']})")

    ratio = v7_result["ppl"] / max(base_result["ppl"], 1e-9)
    print(f"\n[smoke] V7/base PPL ratio = {ratio:.2f}×")
    if ratio > 5.0:
        verdict = "FAIL (>5× — V7-reasoning failure mode replicated)"
    elif ratio > 2.0:
        verdict = "WARN (2-5× — specialist trade-off, evaluate intent)"
    else:
        verdict = "OK (<2× — LM substantively preserved)"
    print(f"[smoke] verdict: {verdict}")

    out = {
        "base": args.base,
        "offsets": args.offsets,
        "n_layer_head_pairs": len(layer_head_pairs),
        "modulated_layers": sorted({li for li, _ in layer_head_pairs}),
        "base_ppl": base_result,
        "v7_ppl": v7_result,
        "ratio": ratio,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[smoke] saved: {args.out}")


# -- lm-evaluation-harness delegation ----------------------------------------
def run_lm_eval(args):
    """Delegate to lm_eval.simple_evaluate via the registered v7hf model."""
    from lm_eval import simple_evaluate

    _register_v7hflm()

    model_args_parts = [
        f"base={args.base}",
        f"offsets={args.offsets}",
        f"dtype={args.dtype}",
    ]
    # Gemma 4 nested text_config makes HFLM fall back to max_length=2048
    # (memory: feedback_lm_eval_2048). Explicit override flows through V7HFLM
    # **kwargs into HFLM.__init__(max_length=...).
    if args.max_length:
        model_args_parts.append(f"max_length={args.max_length}")
    if args.null_adapter:
        model_args_parts.append("null_adapter=True")
    if args.layer_lo is not None:
        model_args_parts.append(f"layer_lo={args.layer_lo}")
    if args.layer_hi is not None:
        model_args_parts.append(f"layer_hi={args.layer_hi}")
    if abs(float(args.offset_scale) - 1.0) > 1e-9:
        model_args_parts.append(f"offset_scale={args.offset_scale}")
    model_args = ",".join(model_args_parts)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[v7hf] tasks={tasks} limit={args.limit} apply_chat_template={args.apply_chat_template}")

    # Parse gen_kwargs string (e.g., "max_gen_toks=1024,temperature=0.0")
    # Needed for generate_until tasks like MMLU-Pro where default max_gen_toks=2048
    # collides with Gemma 4's 2048 max seq len (observed 2026-05-11).
    gen_kwargs = None
    if args.gen_kwargs:
        gen_kwargs = args.gen_kwargs  # lm-eval simple_evaluate accepts the string form

    # Same for confirm_run_unsafe_code (needed for humaneval gated)
    extra_kwargs = {}
    if args.confirm_run_unsafe_code:
        extra_kwargs["confirm_run_unsafe_code"] = True
    if gen_kwargs:
        extra_kwargs["gen_kwargs"] = gen_kwargs
    if args.log_samples:
        extra_kwargs["log_samples"] = True
    if args.num_fewshot is not None:
        extra_kwargs["num_fewshot"] = args.num_fewshot
    if args.fewshot_as_multiturn:
        if not args.apply_chat_template:
            raise SystemExit(
                "--fewshot-as-multiturn requires --apply-chat-template (lm-eval "
                "renders multi-turn exemplars via the chat template)."
            )
        extra_kwargs["fewshot_as_multiturn"] = True

    t0 = time.time()
    results = simple_evaluate(
        model="v7hf",
        model_args=model_args,
        tasks=tasks,
        limit=(args.limit if args.limit and args.limit > 0 else None),
        apply_chat_template=args.apply_chat_template,
        batch_size=args.batch_size,
        **extra_kwargs,
    )
    elapsed = time.time() - t0
    print(f"[v7hf] eval finished in {elapsed:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[v7hf] saved: {args.out}")

    if "results" in results:
        print("\n[v7hf] === per-task summary ===")
        for task_name, task_res in results["results"].items():
            metrics = {k: v for k, v in task_res.items() if not k.startswith("alias")}
            print(f"  {task_name}: {metrics}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Path or HF id of base model")
    ap.add_argument("--offsets", required=True, help="V7 LoFiT offsets .pt blob")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--ppl-smoke", action="store_true",
                    help="Run LM-preservation PPL smoke (5 sentences) instead of lm-eval")
    ap.add_argument("--tasks", default="arc_challenge,truthfulqa_mc1,hellaswag,winogrande",
                    help="Comma-separated lm-eval task names")
    ap.add_argument("--limit", type=int, default=400,
                    help="Per-task sample limit; 0 = full split (passes limit=None to lm-eval)")
    ap.add_argument("--apply-chat-template", action="store_true", default=False,
                    help="Pass --apply_chat_template to lm-eval (REQUIRED for IT models)")
    ap.add_argument("--batch-size", default="1", help="lm-eval batch size (str or int)")
    ap.add_argument("--num-fewshot", type=int, default=None,
                    help="num_fewshot for simple_evaluate (None = each task's lm-eval "
                         "default). Set 5 for the few-shot base/adapter matrix (pre-reg R2/R6).")
    ap.add_argument("--fewshot-as-multiturn", action="store_true", default=False,
                    help="Render few-shot exemplars as multi-turn chat (strongest IT "
                         "base; requires --apply-chat-template). Pre-reg R2.")
    ap.add_argument("--null-adapter", action="store_true", default=False,
                    help="Zero alpha/theta after loading offsets -> v7hf wrapper is a "
                         "behavioral identity. Proves the lift is the offsets, not the "
                         "wrapper codepath. Pre-reg R1/R7.")
    ap.add_argument("--layer-lo", type=int, default=None,
                    help="Causal localization: keep only adapter heads in layers >= layer-lo "
                         "(layer-band patching of theta_mc). Default None = all layers "
                         "(canonical d_mc).")
    ap.add_argument("--layer-hi", type=int, default=None,
                    help="Causal localization: keep only adapter heads in layers <= layer-hi.")
    ap.add_argument("--offset-scale", type=float, default=1.0,
                    help="Activation-scale strength sweep (P-dose): multiply the trained "
                         "offset magnitude by this factor. 1.0=canonical (+35 headline acc_norm). "
                         "0.0=exact base (alpha*0; run --offset-scale 0 == base as first sanity). "
                         "Same axis as score_mc_premise_pod.py so the acc_norm dose-curve aligns "
                         "with the DC-PMI + gsm8k activation-scale curves for the P-dose GATE.")
    ap.add_argument("--max-length", type=int, default=None,
                    help="Override HFLM max_length (Gemma 4 nested config falls back "
                         "to 2048 silently — pass 4096+ for few-shot gen tasks like gsm8k).")
    ap.add_argument("--gen-kwargs", default=None,
                    help="Comma-separated kwargs for generate_until tasks "
                         "(e.g., 'max_gen_toks=1024'). Required for MMLU-Pro and "
                         "similar gen tasks on Gemma 4 (2048 default exceeds model "
                         "max seq len). Passed to simple_evaluate's gen_kwargs param.")
    ap.add_argument("--confirm-run-unsafe-code", action="store_true", default=False,
                    help="Required for humaneval and other code-eval tasks.")
    ap.add_argument("--log-samples", action="store_true", default=True,
                    help="Populate per-sample data in results['samples'] (default "
                         "True; needed for Phase B1 calibration on base run). "
                         "Pass --no-log-samples to disable.")
    ap.add_argument("--no-log-samples", dest="log_samples", action="store_false",
                    help="Disable per-sample logging (smaller output JSON).")
    args = ap.parse_args()

    if args.ppl_smoke:
        run_ppl_smoke(args)
    else:
        run_lm_eval(args)


if __name__ == "__main__":
    main()
