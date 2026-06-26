#!/usr/bin/env python
"""P2 — GPQA cold-MC lift stratified by arbiter: base gen-CoT correctness per item,
ALIGNED to the cold-MC GPQA doc_ids. POD / GPU.

WHY (handoff 2026-06-26 §UPDATE pendiente #1; ledger B7f)
--------------------------------------------------------
The user's hypothesis: the adapter is a System-1 readout that rescues answers the
model already knows, and "does NOT replicate on genuinely hard questions". The EUR0
ClaraVT cut (analyze_lift_difficulty_stratification) found the fix is recoverability-
gated (base-wrong & CoT-right 0.785 vs CoT-wrong 0.143, burial-matched) but ARC has
only ~7 CoT-wrong items (base CoT ~0.97) -> underpowered. GPQA (base CoT 0.70, ~60
genuinely-hard items) gives the POWER to confirm "no replication on hard".

This is the GPQA twin of gen_arbiter_arc_cot_pod.py: it produces
`base_cot_correct` per cold-MC doc_id so analyze_lift_difficulty_stratification can
stratify the GPQA lift by CoT-accessibility:
  --task leaderboard_gpqa_diamond --null d_base_gpqa.json --mc d_mc_gpqa.json
  --arbiter arbiter_gpqa_cot.json

ALIGNMENT (drift-proof, the same move as P1)
--------------------------------------------
GPQA enumerate-ids drift across HF revisions ([[feedback_gpqa_enumerate_id_instability]]).
So by DEFAULT we derive the arbiter items from the cold-MC run itself
(--from-coldmc d_base_gpqa.json): each sample carries the raw GPQA `doc` AND the
lm-eval `doc_id`, so the CoT correctness joins to the cold-MC scoring with zero
enumerate-drift. (--from-dataset loads Idavidrein/gpqa with a PINNED --revision as a
fallback; then doc_id == dataset index, which only aligns if lm-eval used the same
revision.)

SELF-CONSISTENCY@k (cleans the unparsed + stabilises the hard cell)
------------------------------------------------------------------
--k 1 (default) = greedy, one trace (matches the ARC arbiter). --k>1 = sample k
traces (temperature) and majority-vote the parsed letter; an item is correct iff the
vote equals gold. This recovers items where a single greedy trace failed to parse and
de-noises the load-bearing CoT-wrong cell. Reuses build_prompt/parse_answer from
eval_gpqa_cot_fast so the CoT scaffold matches the canonical base-CoT 0.70.

Usage (pod, GPU; cd /workspace/MSAP per block):
  export BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' | grep -v /.locks/ | head -1)")
  # preferred: align to the cold-MC GPQA run (R10b of eval-bulletproof, or a standalone run)
  python -m scripts.gen_arbiter_gpqa_cot_pod --base "$BASE" \
      --from-coldmc runs/eval_bulletproof/R10_base_gpqa --k 1 \
      --out runs/vinf_causal/arbiter_gpqa_cot.json
  # self-consistency@5:  --k 5
Compile-check: python -c "import ast;ast.parse(open('scripts/gen_arbiter_gpqa_cot_pod.py').read())"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

# Reuse the canonical GPQA CoT scaffold so this matches the base-CoT 0.70 measurement.
from scripts.eval_gpqa_cot_fast import (
    build_prompt, parse_answer, build_messages, render, THINK_MARKERS,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def normalize_gpqa_doc(doc: dict) -> dict:
    """Return a dict with the keys build_prompt expects (Question / Correct Answer /
    Incorrect Answer 1-3), accepting either the raw Idavidrein/gpqa row (lm-eval
    --log-samples stores the original doc) or a couple of processed fallbacks."""
    if "Correct Answer" in doc and "Incorrect Answer 1" in doc:
        return doc
    q = doc.get("Question") or doc.get("question")
    # processed fallback: choices list + answer index/letter
    choices = doc.get("choices") or doc.get("options")
    if q and isinstance(choices, (list, tuple)) and len(choices) == 4:
        ans = doc.get("answer", doc.get("gold", doc.get("label")))
        if isinstance(ans, str) and ans in "ABCD":
            ans = "ABCD".index(ans)
        ans = int(ans)
        inc = [c for i, c in enumerate(choices) if i != ans]
        return {"Question": q, "Correct Answer": choices[ans],
                "Incorrect Answer 1": inc[0], "Incorrect Answer 2": inc[1],
                "Incorrect Answer 3": inc[2]}
    raise KeyError(f"cannot read GPQA fields from doc keys={list(doc.keys())[:12]} — "
                   f"adjust normalize_gpqa_doc for this format.")


def load_items(args):
    """Return list of {doc_id, prompt, gold} preserving the cold-MC doc_id."""
    if args.from_coldmc:
        p = Path(args.from_coldmc)
        # lm-eval --output_path produces a dir or a file; accept both
        jpath = p
        if p.is_dir():
            cands = sorted(p.glob("**/*.json"))
            if not cands:
                raise FileNotFoundError(f"no JSON under {p}")
            jpath = cands[0]
        d = json.load(open(jpath, encoding="utf-8"))
        samples = None
        if "samples" in d:
            key = args.task if args.task in d["samples"] else next(iter(d["samples"]))
            samples = d["samples"][key]
        elif isinstance(d, list):
            samples = d
        if samples is None:
            raise KeyError(f"{jpath.name}: no 'samples'; keys={list(d.keys())[:8]}")
        print(f"[arb-gpqa] from-coldmc {jpath.name}: {len(samples)} samples (doc_id-aligned)",
              flush=True)
        items = []
        for s in samples:
            ndoc = normalize_gpqa_doc(s["doc"])
            prompt, gold = build_prompt(ndoc)
            items.append({"doc_id": int(s["doc_id"]), "prompt": prompt, "gold": gold})
        return items
    # fallback: dataset (PIN --revision for stable doc_id)
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train",
                      revision=args.revision)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[arb-gpqa] from-dataset Idavidrein/gpqa@{args.revision}: {len(ds)} items "
          f"(doc_id = dataset index)", flush=True)
    items = []
    for i, doc in enumerate(ds):
        prompt, gold = build_prompt(doc)
        items.append({"doc_id": i, "prompt": prompt, "gold": gold})
    return items


def majority_letter(letters):
    cand = [x for x in letters if x is not None]
    if not cand:
        return None
    return Counter(cand).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--from-coldmc", default=None,
                    help="cold-MC GPQA lm-eval output (dir or json) to align doc_ids (PREFERRED)")
    ap.add_argument("--task", default="leaderboard_gpqa_diamond")
    ap.add_argument("--revision", default=None, help="HF dataset revision (PIN for from-dataset)")
    ap.add_argument("--k", type=int, default=1, help="self-consistency samples (1 = greedy)")
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="8192 — 2048 truncates 96%% of CoT on Gemma 4 (handoff)")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16, help="HF fallback batch size")
    ap.add_argument("--gpu-mem-util", type=float, default=0.88)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/vinf_causal/arbiter_gpqa_cot.json")
    args = ap.parse_args()
    enable_thinking = not args.no_thinking

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = load_items(args)
    if args.limit and not args.from_coldmc:
        pass  # already limited in load_items
    elif args.limit:
        items = items[:args.limit]
    rendered = [render(tokenizer, build_messages(it["prompt"], enable_thinking),
                       enable_thinking)[0] for it in items]
    think_in = any(mk in rendered[0] for mk in THINK_MARKERS)
    print(f"[arb-gpqa] N={len(items)} k={args.k} thinking_in_input={think_in} "
          f"max_new={args.max_new_tokens}", flush=True)
    if enable_thinking and not think_in:
        print("[WARN] no thinking scaffold in rendered prompt — base may NOT reason; "
              "the arbiter would measure base WITHOUT CoT. Inspect the template.", flush=True)

    t0 = time.time()
    # per-item list of k decoded strings (ORIGINAL item order)
    samples_text = _generate(args, rendered, tokenizer)

    results, n_correct, n_unparsed = [], 0, 0
    for it, traces in zip(items, samples_text):
        letters = [parse_answer(t) for t in traces]
        vote = majority_letter(letters)
        ok = int(vote is not None and vote == it["gold"])
        if vote is None:
            n_unparsed += 1
        n_correct += ok
        results.append({"doc_id": it["doc_id"], "base_cot_correct": ok,
                        "pred_letter": vote, "gold_letter": it["gold"],
                        "votes": letters, "k": args.k})
    n = len(results)
    summary = {"task": args.task, "n": n, "k": args.k,
               "base_cot_acc_all": n_correct / max(n, 1),
               "n_unparsed": n_unparsed, "max_new_tokens": args.max_new_tokens,
               "from_coldmc": args.from_coldmc, "elapsed_sec": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[arb-gpqa] base_cot_acc={summary['base_cot_acc_all']:.4f} unparsed={n_unparsed} "
          f"-> {args.out}", flush=True)
    print("READ: low base GPQA-CoT acc => genuinely-hard headroom. Stratify the cold-MC "
          "GPQA lift by this with analyze_lift_difficulty_stratification --arbiter.", flush=True)


def _generate(args, rendered, tokenizer):
    """Return list[list[str]] — k decoded traces per prompt, original order."""
    n, k = len(rendered), max(1, args.k)
    greedy = (k == 1)
    # ---- vLLM (primary) ----
    if not args.no_vllm:
        try:
            from vllm import LLM, SamplingParams
            print(f"[vllm] init (mem={args.gpu_mem_util} max_len={args.max_model_len}) ...", flush=True)
            llm = LLM(model=args.base, dtype="bfloat16",
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len, seed=args.seed)
            if greedy:
                sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed)
            else:
                sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, n=k,
                                    max_tokens=args.max_new_tokens, seed=args.seed)
            print(f"[vllm] generating {n} prompts x k={k} ...", flush=True)
            outs = llm.generate(rendered, sp)
            return [[o.outputs[j].text for j in range(len(o.outputs))] for o in outs]
        except Exception as e:  # noqa: BLE001
            print(f"[vllm] unavailable/failed ({type(e).__name__}: {e}); HF fallback.", flush=True)
    # ---- batched HF (fallback) ----
    import torch
    from transformers import AutoModelForCausalLM
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tokenizer.padding_side = "left"
    gkw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if greedy:
        gkw.update(do_sample=False, num_return_sequences=1)
    else:
        gkw.update(do_sample=True, temperature=1.0, top_p=0.95, top_k=64, num_return_sequences=k)
    order = sorted(range(n), key=lambda i: len(rendered[i]))
    out = [None] * n
    bs = args.batch_size
    t0 = time.time()
    for b in range(0, n, bs):
        idxs = order[b:b + bs]
        enc = tokenizer([rendered[i] for i in idxs], return_tensors="pt",
                        padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, **gkw)
        in_len = enc.input_ids.shape[1]
        gen = gen.view(len(idxs), k, -1)
        for kk, i in enumerate(idxs):
            traces = []
            for j in range(k):
                new = gen[kk, j, in_len:].tolist()
                while new and new[-1] == tokenizer.pad_token_id:
                    new.pop()
                traces.append(tokenizer.decode(new, skip_special_tokens=True))
            out[i] = traces
        done = min(b + bs, n)
        print(f"  [hf {done}/{n}] {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    main()
