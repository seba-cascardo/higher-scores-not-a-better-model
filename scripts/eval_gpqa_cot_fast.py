#!/usr/bin/env python
"""GPQA Diamond CoT-gen arbiter — FAST (vLLM continuous batching) + thinking-ON done right.

WHY THIS EXISTS (replaces the batch=1 path of eval_gpqa_cot_thinking.py for B1):
  The original ran model.generate() one item at a time, greedy, to the full 2048-token
  cap → ~100 s/item → ~560 min for 198 GPQA items. That is the CLAUDE.md gotcha: batch=1
  autoregressive decode is memory-bandwidth-bound (GPU ~50% util); batching / vLLM
  amortizes the weight reads. AND it relied ONLY on the enable_thinking kwarg, which did
  NOT inject the thinking token (rendered prompt had no thinking scaffold; thinking_tokens
  check looked for the wrong markers) → it was generating non-CoT rambling to the cap.

WHAT THIS FIXES:
  1. SPEED: vLLM (continuous batching, all prompts in flight) → minutes, not hours. Falls
     back to BATCHED HF generation (left-pad + attention_mask) if vLLM is unavailable or
     does not support this arch — still ~10-20x over batch=1, and fixes the attention-mask
     warning. Both paths share build_prompt / parse_answer / schema.
  2. THINKING-ON (model card): "thinking is enabled by including the <|think|> token at the
     start of the SYSTEM prompt." We add a system message with THINK_TOKEN (the documented
     enabler) AND pass enable_thinking=True (belt-and-suspenders), then VERIFY the rendered
     prompt actually contains a thinking scaffold and warn loudly if not. We also detect the
     thinking trace (<|channel>thought ... <channel|>) in the OUTPUT and report thinking_rate.
  3. EARLY TERMINATION: with thinking ON + the real chat template, the model emits its
     turn-end token and stops (instead of rambling to 2048) → most items finish well under
     the cap, so wall-clock drops further. vLLM stops on the model's generation_config EOS.

This is the B1 ARBITER: does the BASE model SOLVE GPQA with CoT? On ARC the base is near
CoT-ceiling (mc_only 123/124), so the off-axis "shortcut to CoT-reachable knowledge" claim
has little power there. GPQA is where the base FAILS with CoT → real headroom for the
verifier-guided arm. Low base GPQA-CoT acc => headroom (good for the arbiter); high => the
no-transfer / "knowledge the base already reaches" story holds even on hard tasks.

Usage (pod, GPU; cd /workspace/MSAP per block):
  pip install -q vllm   # if not present; falls back to batched HF if it fails/unsupported
  python scripts/eval_gpqa_cot_fast.py --model "$BASE" \
      --out runs/phase1/b1_gpqa_base_cot.json --limit 200 --greedy --seed 0
  # force the HF batched path (skip vLLM):  --no-vllm --batch-size 24
Compile-check:  python -c "import ast;ast.parse(open('scripts/eval_gpqa_cot_fast.py').read())"
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Model-card enabler: "<|think|>" at the START of the system prompt turns reasoning on.
THINK_TOKEN = "<|think|>"
# Markers of an actual thinking scaffold (input render OR output trace). Gemma 4 uses the
# <|channel>thought ... <channel|> family (see score_mc_kappa_pod.py + the HF model card),
# NOT <think>/<start_of_turn>think. We check a superset so a template tweak still detects it.
THINK_MARKERS = ("<|think|>", "<|channel>", "<channel|>", "<think>")  # NO bare "thought" (false-pos on "I thought")


def build_prompt(item, seed_offset=0):
    """GPQA Diamond chat prompt with 4 shuffled choices. Deterministic shuffle by item hash."""
    rng = random.Random(hash(item["Question"]) + seed_offset)
    choices = [item["Incorrect Answer 1"], item["Incorrect Answer 2"],
               item["Incorrect Answer 3"], item["Correct Answer"]]
    rng.shuffle(choices)
    correct_letter = chr(65 + choices.index(item["Correct Answer"]))
    prompt = (
        f"What is the correct answer to this question: {item['Question']}\n\n"
        f"Choices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n\n"
        "Think step by step, then state your final answer as: "
        "'The answer is (X)' where X is one of A, B, C, D."
    )
    return prompt, correct_letter


def parse_answer(response):
    """Extract A/B/C/D. Priority: explicit 'answer is (X)' > '(X)' in tail > lone X in tail.
    Takes the LAST match so a thinking trace mentioning letters does not fool it."""
    m = re.findall(r"answer\s*(?:is|:)\s*\(?([ABCD])\)?", response, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\(([ABCD])\)", response[-300:])
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\b", response[-100:])
    if m:
        return m[-1]
    return None


def build_messages(prompt, enable_thinking):
    """System <|think|> (documented enabler) + user. Gemma 4 supports the system role."""
    if enable_thinking:
        return [{"role": "system", "content": THINK_TOKEN},
                {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]


def render(tokenizer, messages, enable_thinking):
    """apply_chat_template -> string (add_generation_prompt). Tries the enable_thinking
    kwarg; falls back if the template does not accept it. Returns (text, used_kwarg)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking), True
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-thinking", action="store_true", help="ablation: disable thinking")
    ap.add_argument("--greedy", action="store_true",
                    help="greedy (default: t=1.0 top_p=0.95 top_k=64)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vllm", action="store_true", help="force the batched-HF path")
    ap.add_argument("--batch-size", type=int, default=24, help="HF fallback batch size")
    ap.add_argument("--gpu-mem-util", type=float, default=0.88, help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=8192, help="vLLM max_model_len")
    args = ap.parse_args()
    enable_thinking = not args.no_thinking

    print(f"[setup] model={args.model} thinking={enable_thinking} "
          f"decoding={'greedy' if args.greedy else 'sample'} max_new={args.max_new_tokens} "
          f"seed={args.seed}", flush=True)

    print("[load] tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[data] GPQA Diamond train split ...", flush=True)
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    prompts, golds = [], []
    for item in ds:
        p, g = build_prompt(item)
        prompts.append(render(tokenizer, build_messages(p, enable_thinking), enable_thinking)[0])
        golds.append(g)
    print(f"[data] N={len(prompts)}", flush=True)

    # Thinking sanity on the INPUT render (warn loudly if the scaffold is missing).
    dbg = prompts[0]
    think_in_input = any(mk in dbg for mk in THINK_MARKERS)
    print(f"[debug] last 400 chars of rendered prompt:\n...{dbg[-400:]}", flush=True)
    if enable_thinking and not think_in_input:
        print("[WARN] thinking requested but NO thinking scaffold in the rendered prompt "
              f"(checked {THINK_MARKERS}). The model may NOT reason — B1 would measure base "
              "WITHOUT CoT. Inspect the chat template / system <|think|> handling.", flush=True)
    else:
        print(f"[debug] thinking_scaffold_in_input={think_in_input}", flush=True)

    t0 = time.time()
    backend = None
    gen_ids = None      # list[list[int]] of GENERATED token ids per item, ORIGINAL order

    # ---- vLLM path (primary) ------------------------------------------------
    if not args.no_vllm:
        try:
            from vllm import LLM, SamplingParams
            print(f"[vllm] init (gpu_mem_util={args.gpu_mem_util} max_len={args.max_model_len}) ...",
                  flush=True)
            llm = LLM(model=args.model, dtype="bfloat16",
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len, seed=args.seed)
            if args.greedy:
                sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed)
            else:
                sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                                    max_tokens=args.max_new_tokens, seed=args.seed)
            print(f"[vllm] generating {len(prompts)} prompts (continuous batching) ...", flush=True)
            outs = llm.generate(prompts, sp)                    # vLLM keeps input order
            gen_ids = [list(o.outputs[0].token_ids) for o in outs]
            backend = "vllm"
        except Exception as e:  # noqa: BLE001 — unsupported arch / OOM / not installed
            print(f"[vllm] unavailable/failed ({type(e).__name__}: {e}); falling back to batched HF.",
                  flush=True)

    # ---- batched HF path (fallback; left-pad + attention_mask) --------------
    if gen_ids is None:
        import torch
        from transformers import AutoModelForCausalLM
        torch.manual_seed(args.seed)
        print("[hf] loading model (bf16 cuda) ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        tokenizer.padding_side = "left"                         # left-pad for batched gen
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        gen_kwargs.update(dict(do_sample=False) if args.greedy
                          else dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64))
        # sort by length to minimize padding, remember order to restore
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        gen_ids = [None] * len(prompts)
        bs = args.batch_size
        for b in range(0, len(order), bs):
            idxs = order[b:b + bs]
            enc = tokenizer([prompts[i] for i in idxs], return_tensors="pt",
                            padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, **gen_kwargs)
            in_len = enc.input_ids.shape[1]                     # batch is left-padded -> uniform
            for k, i in enumerate(idxs):
                new = gen[k][in_len:].tolist()                  # generated tokens only
                while new and new[-1] == tokenizer.pad_token_id:  # trim trailing right-pad
                    new.pop()
                gen_ids[i] = new
            done = min(b + bs, len(order))
            el = time.time() - t0
            print(f"  [hf {done}/{len(order)}] {el:.0f}s rate={done/max(el,1e-9):.2f}it/s", flush=True)
        backend = "hf_batched"

    # ---- score (uniform over both backends: parse on CLEAN, detect thinking on FULL) ----
    results, correct_count, think_out = [], 0, 0
    for i, (ids, gold) in enumerate(zip(gen_ids, golds)):
        resp_clean = tokenizer.decode(ids, skip_special_tokens=True)   # parse answer here
        resp_full = tokenizer.decode(ids, skip_special_tokens=False)   # detect <|channel>thought here
        pred = parse_answer(resp_clean)
        correct = (pred == gold)
        correct_count += int(correct)
        has_think = any(mk in resp_full for mk in THINK_MARKERS)
        think_out += int(has_think)
        results.append({"idx": i, "gold": gold, "pred": pred, "correct": correct,
                        "output_tokens": len(ids), "thinking_in_output": has_think,
                        "response": resp_clean})
    n = len(results)
    acc = correct_count / n if n else 0.0
    parse_fail = sum(1 for r in results if r["pred"] is None)
    avg_out = sum(r["output_tokens"] for r in results) / n if n else 0.0
    think_rate = think_out / n if n else 0.0
    elapsed = time.time() - t0

    summary = {
        "model": args.model, "task": "gpqa_diamond_cot_thinking", "backend": backend,
        "n": n, "correct": correct_count, "acc": acc, "parse_fail": parse_fail,
        "avg_output_tokens": avg_out, "thinking_rate_output": think_rate,
        "thinking_scaffold_in_input": think_in_input, "elapsed_sec": elapsed,
        "args": vars(args), "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] backend={backend} acc={acc:.4f} ({correct_count}/{n})  "
          f"parse_fail={parse_fail} ({100*parse_fail/max(n,1):.1f}%)", flush=True)
    print(f"[done] thinking_rate(output)={think_rate:.3f}  avg_out_tokens={avg_out:.0f}  "
          f"wall={elapsed/60:.1f}min", flush=True)
    if enable_thinking and think_rate < 0.5:
        print("[WARN] thinking_rate < 0.5 in OUTPUT — the model mostly did NOT emit a reasoning "
              "trace. Treat acc as base-WITHOUT-CoT and fix thinking enablement before using B1 "
              "as the CoT arbiter.", flush=True)
    print(f"[done] saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
