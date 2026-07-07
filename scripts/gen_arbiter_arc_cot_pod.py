"""ARC arbiter — base gen-CoT correctness per item (pod, GPU).

For the off-axis ARC question: of the 124 items the mc adapter "fixes" in cold MC
scoring but the on-axis v_inf offset does NOT (the mc_only set), does the BASE model
already get them right WITH generative chain-of-thought?
  base-CoT-right  => adapter recovers latent-but-known knowledge (real-ish)
  base-CoT-wrong  => adapter's cold-scoring "fix" is unrelated to knowledge (off-axis)

Reads runs/vinf_causal/arbiter_arc_sets.json (self-contained items: question, labeled
choices, gold). Generates greedy CoT, parses the final letter, exact_match vs gold.
Writes per-item {doc_id, arc_id, base_cot_correct, pred_letter, gold_letter} so the
local analyzer joins to the mc_only/fixed_by_both/vinf_only sets.

Self-contained (no dataset reload => same 400-slice as B). Progress logged per batch.

Usage (pod):
  cd /workspace/MSAP && pwd
  export BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' | grep -v /.locks/ | head -1)")
  python scripts/gen_arbiter_arc_cot_pod.py --base "$BASE" --max-new-tokens 1024 --batch-size 16
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SETS = "runs/vinf_causal/arbiter_arc_sets.json"
OUT = "runs/vinf_causal/arbiter_arc_cot.json"
# stop ids include <end_of_turn>=106 (Gemma 4 turn close) — list is correct, earlier comment was stale
STOP_IDS = [1, 50, 105, 106]


def build_prompt(item):
    q = item["question"]
    ch = item["choices"]
    labels = ch["label"]
    texts = ch["text"]
    lines = [f"{lab}. {txt}" for lab, txt in zip(labels, texts)]
    body = (f"Question: {q}\n" + "\n".join(lines) +
            "\n\nThink step by step, then end your answer with 'Answer: <letter>'.")
    return body, labels


def gold_letter(item):
    ch = item["choices"]
    g = item.get("gold")
    if g is None:
        g = item.get("target")
    try:
        return ch["label"][int(g)]
    except Exception:
        # answerKey fallback (already a letter or 1-based)
        ak = item.get("answerKey")
        if isinstance(ak, str) and ak in ch["label"]:
            return ak
        return None


def parse_letter(text, labels):
    # labels can be letters (A-D...) or numeric strings ("1"-"4") — build the class dynamically
    lab_set = {l.upper() for l in labels}
    alts = "|".join(re.escape(l) for l in sorted(lab_set))
    # prefer explicit "Answer: X" (letter or number)
    m = list(re.finditer(rf"[Aa]nswer\s*[:\-]?\s*\(?({alts})\)?", text, re.IGNORECASE))
    if m:
        return m[-1].group(1).upper()
    # else last standalone option token present in labels.
    # UPPERCASE-only for letters (avoids matching the English article "a" as option A);
    # numeric labels are unambiguous.
    cand = [c for c in re.findall(rf"\b({alts})\b", text) if c.upper() in lab_set]
    if cand:
        return cand[-1].upper()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 400")
    ap.add_argument("--subset", default="all", choices=["all", "base_wrong", "mc_only"],
                    help="P7/TTC: 'base_wrong' = the 198 items the base fails in cold-MC "
                         "(complement of sets['base_right']) — the arbiter/TTC population; "
                         "'mc_only' = the 124 adapter-fixed-not-vinf set. Avoids paying k× "
                         "generation on base_right items the TTC baseline does not need.")
    ap.add_argument("--k", type=int, default=1,
                    help="self-consistency samples (1 = greedy, the canonical arbiter; "
                         ">1 majority-votes k sampled traces to clean unparsed + stabilise B7f)")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temp when k>1")
    ap.add_argument("--out", default=OUT,
                    help="output json (default OVERWRITES the greedy arbiter; pass a new "
                         "path for the self-consistency re-grade so the greedy one survives)")
    args = ap.parse_args()

    data = json.load(open(SETS, encoding="utf-8"))
    items = data["items_all"]
    mc_only = set(data["sets"]["mc_only"])
    # --subset filter (doc_id is a string in items_all; sets hold ints — normalize to str)
    if args.subset != "all":
        if args.subset == "base_wrong":
            base_right = {str(x) for x in data["sets"]["base_right"]}
            keep = [it for it in items if str(it["doc_id"]) not in base_right]
        else:  # mc_only
            mc_ids = {str(x) for x in data["sets"]["mc_only"]}
            keep = [it for it in items if str(it["doc_id"]) in mc_ids]
        print(f"[arb] --subset {args.subset}: {len(keep)}/{len(items)} items kept", flush=True)
        items = keep
    if args.limit:
        items = items[:args.limit]
    print(f"[arb] {len(items)} items ({len(mc_only)} mc_only total); "
          f"subset={args.subset} k={args.k} T={args.temperature}; base={args.base}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = []
    n_correct = n_mc_only = n_mc_only_correct = n_unparsed = 0
    t0 = time.time()
    for bi in range(0, len(items), args.batch_size):
        batch = items[bi:bi + args.batch_size]
        prompts, labels_list = [], []
        for it in batch:
            body, labels = build_prompt(it)
            msgs = [{"role": "user", "content": body}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            labels_list.append(labels)
        # FIX 2026-06-29: `enc` was referenced at model.generate (below) but never
        # built — the k>1 self-consistency path had never executed. apply_chat_template
        # already emitted the special tokens (BOS/turn structure), so tokenize the
        # rendered strings WITHOUT adding them again (add_special_tokens=False).
        # padding_side='left' (set above) keeps in_len uniform for the gen[:, :, in_len:] slice.
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        k = max(1, args.k)
        gkw = dict(max_new_tokens=args.max_new_tokens, eos_token_id=STOP_IDS,
                   pad_token_id=tok.pad_token_id)
        if k == 1:
            gkw.update(do_sample=False, num_return_sequences=1)
        else:
            gkw.update(do_sample=True, temperature=args.temperature, top_p=0.95,
                       top_k=64, num_return_sequences=k)
        with torch.no_grad():
            gen = model.generate(**enc, **gkw)
        in_len = enc["input_ids"].shape[1]
        gen = gen.view(len(batch), k, -1)
        for j, it in enumerate(batch):
            letters, glen = [], 0
            decoded_texts = []
            for kk in range(k):
                new = gen[j, kk, in_len:]
                dec = tok.decode(new, skip_special_tokens=True)
                decoded_texts.append(dec)
                letters.append(parse_letter(dec, labels_list[j]))
                glen = max(glen, int(new.shape[0]))
            cand = [x for x in letters if x is not None]
            pl = Counter(cand).most_common(1)[0][0] if cand else None  # majority vote
            gl = gold_letter(it)
            ok = int(pl is not None and gl is not None and pl == gl)
            if pl is None:
                n_unparsed += 1
            n_correct += ok
            did = it["doc_id"]
            if did in mc_only:
                n_mc_only += 1; n_mc_only_correct += ok
            # persist generated text so the parse is auditable post-hoc (last 4000 chars
            # for the greedy k=1 arbiter; per-sample truncated list when k>1).
            gen_text = (decoded_texts[0][-4000:] if k == 1
                        else [t[-2000:] for t in decoded_texts])
            results.append({"doc_id": did, "arc_id": it.get("arc_id"),
                            "base_cot_correct": ok, "pred_letter": pl, "gold_letter": gl,
                            "votes": (letters if k > 1 else None),
                            "in_mc_only": did in mc_only, "gen_len": glen,
                            "gen_text": gen_text})
        done = bi + len(batch)
        print(f"[arb] [{done}/{len(items)}] acc_so_far={n_correct/done:.3f} "
              f"mc_only_acc={ (n_mc_only_correct/n_mc_only) if n_mc_only else 0:.3f} "
              f"unparsed={n_unparsed} ({time.time()-t0:.0f}s)", flush=True)

    summary = {"n": len(items), "base_cot_acc_all": n_correct / max(len(items), 1),
               "n_mc_only": n_mc_only, "mc_only_base_cot_correct": n_mc_only_correct,
               "mc_only_base_cot_acc": (n_mc_only_correct / n_mc_only) if n_mc_only else None,
               "n_unparsed": n_unparsed, "max_new_tokens": args.max_new_tokens, "k": args.k}
    json.dump({"summary": summary, "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"[arb] wrote {args.out}")
    print("READ: mc_only_base_cot_acc HIGH (base already knows w/ CoT) -> adapter recovers "
          "latent knowledge (ARC real-ish). LOW (base fails even w/ CoT) -> cold-scoring "
          "fix is off-axis. Compare vs base_cot_acc on fixed_by_both / base_right (analyzer).")


if __name__ == "__main__":
    main()
