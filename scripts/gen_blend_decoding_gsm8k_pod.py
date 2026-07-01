"""Blend-decoding ε-sweep on gsm8k — does the off-axis adapter TRANSFER to generation?

SCIENTIFIC QUESTION (handoff 2026-07-01):
  The V7-mc adapter gives a large COLD-MC scoring lift (ARC +35) that is off-axis
  (W_know 6.4%) and does NOT transfer to free generation (gsm8k -25, baked). The user
  asked: what if we run the prompt through the adapter only to READ its signal, then
  AVERAGE it with the baseline numbers as a "slight nudge"? In generation that is exactly
  decoding-time logit blending (a.k.a. proxy-tuning / contrastive decoding):

      logit_comb(token) = (1-ε)·logit_base(token) + ε·logit_adapter(token)        per token

  ε=0 → base; ε=1 → adapter; ε∈(0,1) → the "slight nudge". The prediction (geometric:
  off_par=0% / off_perp=100% → the adapter has ZERO component on the functional axis;
  mechanistic: the lift is des-swamping of a latent contrast between FIXED MC options,
  which has no analogue in free gen) is that NO ε>0 window beats base before the
  EOS/structural collapse. This script tests that empirically so the paper can preempt
  the "did you try decoding-time injection?" reviewer question with a measured negative.

DESIGN (low-risk — reuses model.generate, no manual KV-cache loop):
  Each prompt is DOUBLED in the batch: rows [0:N] are the BASE stream, rows [N:2N] the
  ADAPTER stream. A forward-pre-hook on o_proj adds the LoFiT offset ONLY to rows [N:2N]
  (the canonical install_lofit_hooks math, sliced to the adapter half). A LogitsProcessor
  reads both halves each step, computes comb = (1-ε)·base + ε·adapter, and OVERWRITES both
  halves with comb so argmax picks the SAME token for all 2N rows → both streams advance
  identically (token-wise) while keeping divergent internal states / KV caches.

  Math identity: ε=1 → comb = adapter logits → reproduces the adapter exactly.
                 ε=0 → comb = base logits   → reproduces the base exactly.
  SANITY GATE: ε=0 acc must be HIGH (base solves gsm8k w/ CoT); ε=1 acc must COLLAPSE
  (reproduces the -25 pathology). If ε=0 is already collapsed → harness bug, abort.

Self-contained gsm8k harness (NOT lm-eval): few-shot CoT prompt + '#### <num>' parse.
The canonical -25 is an lm-eval number; here we compare ACC(ε) vs ACC(ε=0) WITHIN this
harness, which is what answers the window question. We do NOT claim to reproduce -25 to
the decimal — only the direction (base high, adapter collapses), which the gate checks.

Usage (pod, RTX PRO 6000 96GB; after bring-up + git pull + `hf auth login`):
  cd /workspace/MSAP && pwd
  export BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' | grep -v /.locks/ | head -1)")
  export OFFSETS=runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt   # from HF msap-a1-trio-b2-planner-2026-06-10
  python scripts/gen_blend_decoding_gsm8k_pod.py --base "$BASE" --offsets "$OFFSETS" \
      --n 200 --batch-size 8 --max-new-tokens 512 \
      --eps-grid 0,0.1,0.2,0.35,0.5,0.75,1.0 \
      --out runs/blend_decoding/gsm8k_eps_sweep.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

# Multimodal fallback (Gemma 4 31B IT registers as *ForImageTextToText in transformers >=5.0)
try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_with_router import _resolve_layers, _get_attention_shape  # noqa: E402
from scripts.run_lm_eval_v7 import _filter_compatible_layers, _load_model_multimodal  # noqa: E402

# Gemma 4 stop ids (NOT <end_of_turn>): from generation_config.eos_token_id (see gen_arbiter_*).
STOP_IDS = [1, 50, 105, 106]

# Mutable state read by the per-layer hook: N = real (un-doubled) batch size of the
# CURRENT generate() call. The hook applies the offset only to rows [N:2N].
_HOOK_STATE = {"N": 0}


# -- Row-selective LoFiT hooks (offset on the adapter half [N:2N] only) -------
def install_blend_hooks(layers, alpha, theta, layer_head_pairs, attn_shape):
    """Mirror install_lofit_hooks but add the offset ONLY to batch rows [N:2N].

    Identical per-coordinate math to eval_with_router.install_lofit_hooks for the
    adapter half; the base half [0:N] is untouched. N is read from _HOOK_STATE at
    call time (set before each generate()).
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
        if not hasattr(layer, "self_attn"):
            print(f"  WARN: layer {layer_idx} has no self_attn, skipping")
            continue
        o_proj = layer.self_attn.o_proj
        target_device = o_proj.weight.device
        target_dtype = o_proj.weight.dtype
        layer_offset = torch.zeros(num_q_heads, head_dim, dtype=target_dtype, device=target_device)
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
                    raise RuntimeError(f"L{li}: o_proj input dim {total} != expected {expected_total}")
                N = _HOOK_STATE["N"]
                if N <= 0 or 2 * N != B:
                    raise RuntimeError(f"L{li}: _HOOK_STATE['N']={N} inconsistent with batch B={B} "
                                       f"(expected B==2N). Did you set _HOOK_STATE before generate()?")
                out = concat.clone()
                out[N:] = out[N:] + offset_flat  # adapter half only
                return (out,) + inputs[1:]
            return hook

        handles.append(o_proj.register_forward_pre_hook(make_hook(layer_idx, layer_offset_flat)))
    return handles


class BlendLogitsProcessor(LogitsProcessor):
    """Overwrite both batch halves with comb=(1-ε)·base+ε·adapter so all 2N rows pick
    the same next token (the blended argmax). Operates on logits (pre-softmax)."""

    def __init__(self, eps: float):
        self.eps = float(eps)

    def __call__(self, input_ids, scores):
        B = scores.shape[0]
        N = B // 2
        base = scores[:N]
        adapter = scores[N:]
        comb = (1.0 - self.eps) * base + self.eps * adapter
        scores = scores.clone()
        scores[:N] = comb
        scores[N:] = comb
        return scores


# -- gsm8k harness -----------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_num(s: str):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_answer(ans_field: str):
    m = re.search(r"####\s*(.+)", ans_field)
    return _norm_num(m.group(1)) if m else None


def parse_pred(gen_text: str):
    # Prefer explicit '#### <num>'; else the last number in the generation.
    m = list(re.finditer(r"####\s*(-?[\d,\.]+)", gen_text))
    if m:
        v = _norm_num(m[-1].group(1))
        if v is not None:
            return v
    nums = _NUM_RE.findall(gen_text)
    return _norm_num(nums[-1]) if nums else None


def build_fewshot(tok, k_shot: int):
    """k_shot CoT exemplars from gsm8k train (with their '#### num' answers)."""
    train = load_dataset("gsm8k", "main", split="train")
    parts = ["Solve the math problem. Show your step-by-step reasoning, then give the "
             "final numeric answer on a new line as '#### <number>'.\n"]
    for i in range(k_shot):
        ex = train[i]
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n")
    return "\n".join(parts)


def build_prompt(tok, fewshot_prefix: str, question: str) -> str:
    body = f"{fewshot_prefix}\nQuestion: {question}\nAnswer:"
    msgs = [{"role": "user", "content": body}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# -- Generation (doubled batch + blend) --------------------------------------
@torch.no_grad()
def generate_blend(model, tok, prompts, eps, max_new_tokens, batch_size):
    """Return list of generated strings + per-item gen length, for a given ε."""
    gens, glens = [], []
    n = len(prompts)
    for bi in range(0, n, batch_size):
        chunk = prompts[bi:bi + batch_size]
        N = len(chunk)
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        # Double the batch: [base rows ; adapter rows]
        input_ids = torch.cat([enc["input_ids"], enc["input_ids"]], dim=0)
        attn = torch.cat([enc["attention_mask"], enc["attention_mask"]], dim=0)
        in_len = input_ids.shape[1]
        _HOOK_STATE["N"] = N  # hook applies offset to rows [N:2N]
        out = model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            eos_token_id=STOP_IDS, pad_token_id=tok.pad_token_id,
            logits_processor=LogitsProcessorList([BlendLogitsProcessor(eps)]),
        )
        new = out[:N, in_len:]  # base half carries the (blended) tokens; halves are identical
        for j in range(N):
            row = new[j]
            # gen length up to first stop (generate pads with pad after stop)
            nonpad = (row != tok.pad_token_id).sum().item()
            glens.append(int(nonpad))
            gens.append(tok.decode(row, skip_special_tokens=True))
        done = bi + N
        print(f"    [eps={eps:.2f}] [{done}/{n}] ({len(chunk)} prompts/batch)", flush=True)
    return gens, glens


def evaluate_eps(model, tok, prompts, golds, eps, max_new_tokens, batch_size):
    t0 = time.time()
    gens, glens = generate_blend(model, tok, prompts, eps, max_new_tokens, batch_size)
    n_correct = n_eos = n_unparsed = 0
    per_item = []
    for gen, glen, gold in zip(gens, glens, golds):
        pred = parse_pred(gen)
        ok = int(pred is not None and gold is not None and abs(pred - gold) < 1e-4)
        eos = int(glen < max_new_tokens)  # terminated by stop before the cap
        if pred is None:
            n_unparsed += 1
        n_correct += ok
        n_eos += eos
        per_item.append({"pred": pred, "gold": gold, "correct": ok, "eos": eos, "gen_len": glen})
    n = len(golds)
    acc = n_correct / max(n, 1)
    eos_rate = n_eos / max(n, 1)
    mean_len = sum(glens) / max(len(glens), 1)
    print(f"  [eps={eps:.2f}] acc={acc:.4f}  eos_rate={eos_rate:.3f}  mean_len={mean_len:.1f}  "
          f"unparsed={n_unparsed}  ({time.time()-t0:.0f}s)", flush=True)
    return {"eps": eps, "acc": acc, "eos_rate": eos_rate, "mean_gen_len": mean_len,
            "n_correct": n_correct, "n_eos": n_eos, "n_unparsed": n_unparsed, "n": n,
            "per_item": per_item}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--offsets", required=True, help="V7-mc LoFiT offsets .pt (layer_head_pairs/alpha/theta)")
    ap.add_argument("--n", type=int, default=200, help="gsm8k test items")
    ap.add_argument("--k-shot", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8, help="real batch (doubled internally -> 2N)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--eps-grid", default="0,0.1,0.2,0.35,0.5,0.75,1.0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default="runs/blend_decoding/gsm8k_eps_sweep.json")
    ap.add_argument("--keep-per-item", action="store_true", help="keep per-item rows in JSON (else summary-only)")
    args = ap.parse_args()

    eps_grid = [float(x) for x in args.eps_grid.split(",") if x.strip() != ""]
    print("=" * 70)
    print("BLEND-DECODING ε-SWEEP on gsm8k — does off-axis adapter transfer to gen?")
    print(f"  base={args.base}\n  offsets={args.offsets}\n  n={args.n} k_shot={args.k_shot} "
          f"max_new_tokens={args.max_new_tokens} eps_grid={eps_grid}")
    print("=" * 70)

    torch_dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = _load_model_multimodal(args.base, torch_dtype)
    model.eval()

    # Load offsets + install row-selective hooks (filtered to Gemma-4-compatible layers)
    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    layer_head_pairs = blob["layer_head_pairs"]
    alpha = blob["alpha"]
    theta = blob["theta"]
    print(f"[blend] {len(layer_head_pairs)} (layer,head) pairs loaded")
    layers = _resolve_layers(model)
    attn_shape = _get_attention_shape(model, layers)
    layer_head_pairs, alpha, theta = _filter_compatible_layers(
        model, tok, attn_shape, layer_head_pairs, alpha, theta)
    handles = install_blend_hooks(layers, alpha, theta, layer_head_pairs, attn_shape)
    print(f"[blend] installed {len(handles)} row-selective hooks on layers "
          f"{sorted({li for li, _ in layer_head_pairs})}")

    # Build gsm8k eval set
    test = load_dataset("gsm8k", "main", split="test")
    items = [test[i] for i in range(min(args.n, len(test)))]
    fewshot = build_fewshot(tok, args.k_shot)
    prompts = [build_prompt(tok, fewshot, it["question"]) for it in items]
    golds = [gold_answer(it["answer"]) for it in items]
    print(f"[blend] built {len(prompts)} gsm8k prompts (k_shot={args.k_shot})")

    results = []
    for eps in eps_grid:
        results.append(evaluate_eps(model, tok, prompts, golds, eps,
                                    args.max_new_tokens, args.batch_size))

    remove = [h for h in handles]
    for h in remove:
        h.remove()

    # -- Sanity gate + verdict ------------------------------------------------
    by_eps = {round(r["eps"], 4): r for r in results}
    acc0 = by_eps.get(0.0, {}).get("acc")
    acc1 = by_eps.get(1.0, {}).get("acc")
    gate = {}
    if acc0 is not None:
        gate["base_solves_gate"] = bool(acc0 >= 0.40)  # base should solve gsm8k w/ CoT
    if acc0 is not None and acc1 is not None:
        gate["collapse_reproduced_gate"] = bool(acc1 <= acc0 - 0.10)  # adapter collapses
    best = max(results, key=lambda r: r["acc"])
    # "Window" = any ε in (0,1) strictly beating base by a meaningful margin
    interior = [r for r in results if 0.0 < r["eps"] < 1.0]
    best_interior = max(interior, key=lambda r: r["acc"]) if interior else None
    window = (best_interior is not None and acc0 is not None
              and best_interior["acc"] > acc0 + 0.02)
    verdict = {
        "acc_eps0_base": acc0,
        "acc_eps1_adapter": acc1,
        "best_eps": best["eps"], "best_acc": best["acc"],
        "best_interior_eps": (best_interior["eps"] if best_interior else None),
        "best_interior_acc": (best_interior["acc"] if best_interior else None),
        "transfer_window_exists": bool(window),
        "gates": gate,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_results = results
    if not args.keep_per_item:
        save_results = [{k: v for k, v in r.items() if k != "per_item"} for r in results]
    payload = {
        "config": {"base": args.base, "offsets": args.offsets, "n": args.n,
                   "k_shot": args.k_shot, "max_new_tokens": args.max_new_tokens,
                   "eps_grid": eps_grid, "batch_size": args.batch_size},
        "results": save_results, "verdict": verdict,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("=" * 70)
    print(f"{'eps':>6s}  {'acc':>7s}  {'eos_rate':>9s}  {'mean_len':>9s}")
    for r in results:
        print(f"{r['eps']:>6.2f}  {r['acc']:>7.4f}  {r['eos_rate']:>9.3f}  {r['mean_gen_len']:>9.1f}")
    print("-" * 70)
    print(json.dumps(verdict, indent=2))
    print()
    if gate.get("base_solves_gate") is False:
        print("!!! SANITY FAIL: ε=0 (base) acc < 0.40 — base should solve gsm8k w/ CoT. "
              "Harness/prompt bug suspected; do NOT trust the sweep.")
    if gate.get("collapse_reproduced_gate") is False:
        print("!!! SANITY WARN: ε=1 (adapter) did NOT collapse vs base — check offsets/hook path.")
    if verdict["transfer_window_exists"]:
        print(">>> WINDOW FOUND: an interior ε beats base by >2pp. Off-axis adapter MAY "
              "partially transfer to gen via blending — investigate (unexpected).")
    else:
        print(">>> NO TRANSFER WINDOW: no interior ε beats base by >2pp. Confirms the "
              "off-axis lift does not transfer to generation via decoding-time blend.")
    print(f"[blend] saved: {out_path}")


if __name__ == "__main__":
    main()
