#!/usr/bin/env python
"""Generic MCQ arbiter (HellaSwag / Winogrande) — base gen-CoT correctness per item,
ALIGNED to the cold-MC doc_ids. POD / GPU.

WHY (the 2nd anchor of "out-of-task != hard")
----------------------------------------------------------------------------------------
The within-format control (B8c, ARC-Easy vs ARC-Challenge) showed the cold-MC lift's
fix_rate moves little with difficulty INSIDE ARC (0.86 vs 0.76) while GPQA out-of-task
collapses (0.20). To CLOSE "out-of-task != hard" we also need an independent base-CoT
anchor for HSwag/Wino (lift +17/+19): are those tasks CoT-resolvable by the base (like
ARC 0.97 / GPQA 0.70), so the no-transfer is about distribution, not difficulty?

This is the HSwag/Wino twin of gen_arbiter_arc/gpqa_cot_pod.py. It produces
`base_cot_correct` per cold-MC doc_id so analyze_lift_difficulty_stratification can
stratify the lift by CoT-accessibility (Q2 2x2):
  --task hellaswag  --null d_null_hswag_wino.json --mc d_mc_hswag_wino.json
                    --arbiter arbiter_hswag_cot.json

REFORMULATION (the central caveat)
----------------------------------
HSwag and Wino are NOT letter-MCQ natively (HSwag = pick the best of 4 continuations of
a context; Wino = fill the "_" with one of 2 options). A CoT arbiter that ends in
"Answer: <letter>" needs each item REFORMULATED as a lettered MCQ:
  HSwag -> "Context: {ctx}\n\nA. {e0}\nB. {e1}\nC. {e2}\nD. {e3}\n\nWhich continuation..."
  Wino  -> "Sentence: {sent with _}\n\nA. {opt1}\nB. {opt2}\n\nWhich option fills the blank..."
This reformulation introduces a FORMAT different from the cold-MC scorer (option-continuation,
no letters): the arbiter anchors KNOWLEDGE in letter-format, it does NOT replicate the channel
— the same gap ARC/GPQA also have (where the arbiter still works). Flag it when reading.

ALIGNMENT (drift-proof, same move as the ARC/GPQA arbiters)
----------------------------------------------------------
By default derive items from the cold-MC run itself (--from-coldmc): each lm-eval sample
carries the processed `doc` AND the `doc_id`, so the CoT correctness joins to the cold-MC
scoring with zero re-load drift. (--from-dataset is a fallback; doc_id == dataset index,
aligns only if lm-eval used the same split.)

SELF-CONSISTENCY@k: --k 1 (default) = greedy one trace (matches ARC); --k>1 = sample k
traces and majority-vote the letter (cleans unparsed + de-noises the hard cell).

Usage (pod, GPU; cd /workspace/MSAP per block):
  export BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' | grep -v /.locks/ | head -1)")
  python -m scripts.gen_arbiter_mc_cot_pod --base "$BASE" --task hellaswag \
      --from-coldmc runs/vinf_causal/d_null_hswag_wino.json --max-new-tokens 1024 --k 1 \
      --out runs/vinf_causal/arbiter_hswag_cot.json
  python -m scripts.gen_arbiter_mc_cot_pod --base "$BASE" --task winogrande \
      --from-coldmc runs/vinf_causal/d_null_hswag_wino.json --max-new-tokens 1024 --k 1 \
      --out runs/vinf_causal/arbiter_wino_cot.json
Compile-check: python -c "import ast;ast.parse(open('scripts/gen_arbiter_mc_cot_pod.py').read())"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Gemma 4 stops (NOT <end_of_turn>): from generation_config.eos_token_id.
STOP_IDS = [1, 50, 105, 106]
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# --------------------------------------------------------------------------------------
# per-task field extraction + prompt building. Each returns (body, labels, gold_letter).
# Robust to lm-eval's processed doc (query/choices/gold) AND the raw dataset row.
# --------------------------------------------------------------------------------------
def _hellaswag(doc: dict):
    ctx = (doc.get("query") or doc.get("ctx")
           or (str(doc.get("ctx_a", "")) + " " + str(doc.get("ctx_b", ""))).strip())
    endings = doc.get("choices") or doc.get("endings")
    gold = doc.get("gold", doc.get("label"))
    if endings is None or ctx == "" or gold is None or gold == "":
        raise KeyError(f"hellaswag fields missing (keys={list(doc.keys())[:12]})")
    endings = list(endings)
    gold = int(gold)
    lines = [f"{LABELS[i]}. {e}" for i, e in enumerate(endings)]
    body = (f"Context: {ctx}\n\n" + "\n".join(lines)
            + "\n\nWhich continuation is most plausible? Think step by step, "
              "then end with 'Answer: <letter>'.")
    return body, LABELS[:len(endings)], LABELS[gold]


def _winogrande(doc: dict):
    sent = doc.get("sentence") or doc.get("query")
    o1, o2 = doc.get("option1"), doc.get("option2")
    if (o1 is None or o2 is None) and isinstance(doc.get("choices"), (list, tuple)):
        o1, o2 = doc["choices"][0], doc["choices"][1]
    ans = doc.get("answer", doc.get("gold", doc.get("label")))
    if sent is None or o1 is None or o2 is None or ans is None or ans == "":
        raise KeyError(f"winogrande fields missing (keys={list(doc.keys())[:12]})")
    s = str(ans).strip()
    gi = int(s) - 1 if s in ("1", "2") else int(s)   # answer is '1'/'2'; gold-idx 0/1
    opts = [o1, o2]
    lines = [f"{LABELS[i]}. {o}" for i, o in enumerate(opts)]
    body = (f"Sentence: {sent}\n\n" + "\n".join(lines)
            + "\n\nWhich option best fills the blank '_'? Think step by step, "
              "then end with 'Answer: <letter>'.")
    return body, LABELS[:2], LABELS[gi]


BUILDERS = {"hellaswag": _hellaswag, "winogrande": _winogrande}


def parse_letter(text: str, labels):
    """Prefer explicit 'Answer: X'; else last standalone option letter in `labels`."""
    m = list(re.finditer(r"[Aa]nswer\s*[:\-]?\s*\(?([A-Ha-h])\)?", text))
    if m:
        c = m[-1].group(1).upper()
        return c if c in labels else None
    cand = [c for c in re.findall(r"\b([A-Ha-h])\b", text) if c.upper() in labels]
    return cand[-1].upper() if cand else None


def load_items(args):
    """Return list of {doc_id, body, labels, gold} preserving the cold-MC doc_id."""
    build = BUILDERS[args.task]
    if args.from_coldmc:
        p = Path(args.from_coldmc)
        jpath = p
        if p.is_dir():
            cands = sorted(p.glob("**/*.json"))
            cands = [c for c in cands if "results" in c.name] or cands
            if not cands:
                raise FileNotFoundError(f"no JSON under {p}")
            jpath = cands[0]
        d = json.load(open(jpath, encoding="utf-8"))
        if "samples" not in d:
            raise KeyError(f"{jpath.name}: no 'samples' (keys={list(d.keys())[:8]})")
        key = args.task if args.task in d["samples"] else next(iter(d["samples"]))
        samples = d["samples"][key]
        print(f"[arb-{args.task}] from-coldmc {jpath.name}: {len(samples)} samples "
              f"(task key '{key}', doc_id-aligned)", flush=True)
        items = []
        for s in samples:
            body, labels, gl = build(s["doc"])
            items.append({"doc_id": int(s["doc_id"]), "body": body,
                          "labels": labels, "gold": gl})
        return items
    # fallback: raw dataset (doc_id == index; aligns only if lm-eval used the same split)
    from datasets import load_dataset
    if args.task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
    else:
        ds = load_dataset("winogrande", "winogrande_xl", split="validation",
                          trust_remote_code=True)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[arb-{args.task}] from-dataset: {len(ds)} items (doc_id = index)", flush=True)
    items = []
    for i, doc in enumerate(ds):
        body, labels, gl = build(dict(doc))
        items.append({"doc_id": i, "body": body, "labels": labels, "gold": gl})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--task", required=True, choices=sorted(BUILDERS.keys()))
    ap.add_argument("--from-coldmc", default=None,
                    help="cold-MC lm-eval output (dir or json) to align doc_ids (PREFERRED)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--k", type=int, default=1,
                    help="self-consistency samples (1 = greedy, canonical; >1 majority-votes)")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temp when k>1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_items(args)
    if args.limit and args.from_coldmc:
        items = items[:args.limit]
    if not items:
        raise SystemExit(f"[arb-{args.task}] no items loaded")
    # SAFETY: show the first reformulated prompt + gold so a wrong field-mapping is caught
    # BEFORE burning GPU on all 400 (these scripts are new/unproven for HSwag/Wino).
    print(f"[arb-{args.task}] N={len(items)} k={args.k} max_new={args.max_new_tokens}\n"
          f"---- first prompt (inspect format!) ----\n{items[0]['body']}\n"
          f"---- gold={items[0]['gold']} labels={items[0]['labels']} ----", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda").eval()

    k = max(1, args.k)
    gkw = dict(max_new_tokens=args.max_new_tokens, eos_token_id=STOP_IDS,
               pad_token_id=tok.pad_token_id)
    if k == 1:
        gkw.update(do_sample=False, num_return_sequences=1)
    else:
        gkw.update(do_sample=True, temperature=args.temperature, top_p=0.95,
                   top_k=64, num_return_sequences=k)

    # length-sorted batching to minimise padding (original order restored via idxs)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["body"]))
    results = [None] * len(items)
    n_correct = n_unparsed = 0
    t0 = time.time()
    for b in range(0, len(items), args.batch_size):
        idxs = order[b:b + args.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": items[i]["body"]}],
                                           tokenize=False, add_generation_prompt=True)
                   for i in idxs]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, **gkw)
        in_len = enc["input_ids"].shape[1]
        gen = gen.view(len(idxs), k, -1)
        for kk, i in enumerate(idxs):
            it = items[i]
            letters, glen = [], 0
            for j in range(k):
                new = gen[kk, j, in_len:]
                letters.append(parse_letter(tok.decode(new, skip_special_tokens=True),
                                            it["labels"]))
                glen = max(glen, int(new.shape[0]))
            cand = [x for x in letters if x is not None]
            pl = Counter(cand).most_common(1)[0][0] if cand else None
            ok = int(pl is not None and pl == it["gold"])
            if pl is None:
                n_unparsed += 1
            n_correct += ok
            results[i] = {"doc_id": it["doc_id"], "base_cot_correct": ok,
                          "pred_letter": pl, "gold_letter": it["gold"],
                          "votes": (letters if k > 1 else None), "gen_len": glen}
        done = min(b + args.batch_size, len(items))
        print(f"[arb-{args.task}] [{done}/{len(items)}] "
              f"acc={n_correct/done:.3f} unparsed={n_unparsed} ({time.time()-t0:.0f}s)",
              flush=True)

    summary = {"task": args.task, "n": len(items), "k": args.k,
               "base_cot_acc_all": n_correct / max(len(items), 1),
               "n_unparsed": n_unparsed, "max_new_tokens": args.max_new_tokens,
               "from_coldmc": args.from_coldmc, "elapsed_sec": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[arb-{args.task}] base_cot_acc={summary['base_cot_acc_all']:.4f} "
          f"unparsed={n_unparsed} -> {args.out}", flush=True)
    print("READ: HIGH base CoT-acc (like ARC 0.97) => the +17/+19 rescues CoT-accessible "
          "knowledge => no-transfer is about DISTRIBUTION not difficulty (closes B8c). "
          "LOW => tension with the recoverability-gated frame. Then stratify the cold-MC "
          "lift with analyze_lift_difficulty_stratification --arbiter (2x2 fix x CoT).", flush=True)


if __name__ == "__main__":
    main()
