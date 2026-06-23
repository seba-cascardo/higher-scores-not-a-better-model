"""Exp-A POD pass — answer-only (choices-only) + conditional re-score for the PMI-DC
/ answer-prior test of the +35 MC lift.  GPU / pod.

WHAT THIS PRODUCES (the missing piece for scripts/score_mc_pmi.py):
  For each MC item, the per-option log P(A_i | C_neutro) — the option-text continuation
  log-likelihood scored under a NEUTRAL, question-ablated premise — for BOTH arms
  (base = null offsets, mc = trained offsets).  This is simultaneously:
    * the answer-only term of Holtzman PMI-DC:  score_PMI = logP(A|Q) - logP(A|C_neutro)
    * the choices-only score (Balepur 2024 / Chandak 2024): argmax logP(A|C_neutro)
      = "which option looks like an answer WITHOUT seeing the question".

WHY A SELF-CONTAINED SCRIPT (not an lm-eval YAML variant): run_lm_eval_v7.py has no
--include-path passthrough, and untested YAML shipped to the pod can silently produce
wrong-but-plausible numbers.  Instead this script reuses the SAME proven v7hf hook
path (install_lofit_hooks) and scores option-text continuations directly, and prints
a SANITY GATE: its own CONDITIONAL (question-present) naive lift must reproduce ~+0.35
(ARC) / ~+0.33 (TQA).  If it does, the scaffold is right and the answer-only term is
trustworthy; if not, abort cheaply (this is a short cold-MC pass, no generation).

SCORING (mirrors lm-eval multiple_choice loglik): context built from a user turn,
chat-template-wrapped with add_generation_prompt=True; continuation = " " + option_text;
score = teacher-forced sum log-prob over the continuation tokens.  Items/options/gold
are read from the EXISTING runs/vinf_causal/d_null.json so doc_ids align 1:1 with B.

NOTE: our +35 is on option-TEXT continuation scoring (the lm-eval default for
arc_challenge / truthfulqa_mc1), NOT letter A/B/C/D scoring — so this answer-prior /
PMI test (not a symbol-channel test) is the on-target probe of the off-axis lift.

Usage (pod, GPU; cd /workspace/MSAP per block):
  BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' \
        2>/dev/null | grep -v '/.locks/' | head -1)")
  python scripts/score_mc_premise_pod.py \
      --base "$BASE" --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --task arc_challenge --limit 400 --apply-chat-template \
      --out runs/vinf_causal/answer_only_arc.json
  # then truthfulqa_mc1 the same way (--task truthfulqa_mc1 --out ...answer_only_tqa.json)
  # merge the two per-task files into answer_only_logprobs.json (see --merge below),
  # then LOCAL:  python scripts/score_mc_pmi.py --task arc_challenge --metric acc_norm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.run_lm_eval_v7 import _load_model_multimodal, _filter_compatible_layers
from scripts.eval_with_router import (
    install_lofit_hooks, remove_handles, _resolve_layers, _get_attention_shape,
)
from transformers import AutoTokenizer

# Holtzman domain-conditional neutral premise per task (question text removed, surface
# domain held fixed). Recorded into the output for audit.
NEUTRAL_PREMISE = {
    "arc_challenge": "Answer:",
    "truthfulqa_mc1": "A:",
}
COND_PREMISE = {  # question-present context (mirrors lm-eval doc_to_text shape)
    "arc_challenge": "Question: {q}\nAnswer:",
    "truthfulqa_mc1": "Q: {q}\nA:",
}


def item_fields(sample: dict) -> tuple[str, list[str], int]:
    """(question, option_texts, gold_idx) robust to ARC and TQA mc1 docs."""
    doc = sample["doc"]
    q = doc.get("question") or doc.get("query") or ""
    ch = doc.get("choices")
    if isinstance(ch, dict) and "text" in ch:
        opts = list(ch["text"])
    elif isinstance(ch, list) and ch and isinstance(ch[0], str):
        opts = list(ch)
    elif isinstance(doc.get("mc1_targets"), dict) and "choices" in doc["mc1_targets"]:
        opts = list(doc["mc1_targets"]["choices"])
    else:
        raise KeyError(f"cannot extract options (doc keys {list(doc.keys())})")
    return q, opts, int(sample["target"])


def build_context(tokenizer, premise_str: str, apply_chat_template: bool) -> str:
    if apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": premise_str}],
            tokenize=False, add_generation_prompt=True)
    return premise_str


@torch.no_grad()
def cont_logprob(model, tokenizer, context_str: str, continuation: str) -> float:
    """Sum log-prob of `continuation` tokens given `context_str` (teacher-forced).
    Mirrors lm-eval loglik: tokenize context, tokenize context+continuation, score the
    delta tokens. add_special_tokens=False because the chat template already adds BOS."""
    ctx = tokenizer(context_str, add_special_tokens=False).input_ids
    full = tokenizer(context_str + continuation, add_special_tokens=False).input_ids
    n_ctx = len(ctx)
    if len(full) <= n_ctx:
        return 0.0
    ids = torch.tensor([full], device=model.device)
    logits = model(ids, use_cache=False).logits[0]          # [T, vocab]
    # continuation token at position t (t in [n_ctx, T)) predicted by logits[t-1]
    lp = torch.log_softmax(logits[n_ctx - 1:-1].float(), dim=-1)
    tgt = torch.tensor(full[n_ctx:], device=lp.device)
    return float(lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item())


def score_arm(model, tokenizer, items, task, apply_ct, label):
    """Returns dict doc_id -> {'cond':[lp per opt], 'ao':[lp per opt]} for one arm."""
    cond_tmpl = COND_PREMISE[task]
    neut = NEUTRAL_PREMISE[task]
    out = {}
    t0 = time.time()
    for i, (did, q, opts, gold) in enumerate(items):
        ctx_cond = build_context(tokenizer, cond_tmpl.format(q=q), apply_ct)
        ctx_neut = build_context(tokenizer, neut, apply_ct)
        cond = [cont_logprob(model, tokenizer, ctx_cond, " " + o) for o in opts]
        ao = [cont_logprob(model, tokenizer, ctx_neut, " " + o) for o in opts]
        out[did] = {"cond": cond, "ao": ao, "gold": gold, "n_opt": len(opts)}
        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            el = time.time() - t0
            print(f"  [{label}] {i+1}/{len(items)}  {el:.0f}s  "
                  f"{(i+1)/el:.1f} it/s", flush=True)
    return out


def acc_norm_from(arm_scores, items, key):
    """argmax of length-normalized logprob == gold, mean over items (acc_norm)."""
    hit = []
    for did, q, opts, gold in items:
        lp = np.array(arm_scores[did][key], dtype=np.float64)
        clen = np.array([max(1, len(o)) for o in opts], dtype=np.float64)
        hit.append(int(int(np.argmax(lp / clen)) == gold))
    return float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--offsets", required=True, help="V7-mc offsets .pt")
    ap.add_argument("--task", required=True, choices=list(NEUTRAL_PREMISE))
    ap.add_argument("--cond-run", type=Path, default=Path("runs/vinf_causal/d_null.json"),
                    help="d_null.json — source of the SAME items/options/gold as run B")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out", type=Path, required=True,
                    help="per-task answer-only file (also stores cond for self-consistent PMI)")
    args = ap.parse_args()
    if not (args.base or "").strip():
        raise SystemExit("FATAL: --base empty (fresh pod shell loses $BASE). Re-set it.")

    print("=" * 70, flush=True)
    print(f"Exp-A premise scoring  | task={args.task} limit={args.limit} "
          f"chat_template={args.apply_chat_template}", flush=True)
    print("=" * 70, flush=True)

    # items from d_null.json (same docs/order as B)
    d = json.load(args.cond_run.open(encoding="utf-8"))
    samples = d["samples"][args.task][: args.limit]
    items = []
    for s in samples:
        q, opts, gold = item_fields(s)
        items.append((int(s["doc_id"]), q, opts, gold))
    print(f"  {len(items)} items from {args.cond_run}", flush=True)
    print(f"  neutral premise = {NEUTRAL_PREMISE[args.task]!r}  "
          f"cond template = {COND_PREMISE[args.task]!r}", flush=True)

    torch_dtype = getattr(torch, args.dtype)
    print(f"  loading base {args.base} ...", flush=True)
    t0 = time.time()
    model = _load_model_multimodal(args.base, torch_dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    lhp, alpha, theta = blob["layer_head_pairs"], blob["alpha"], blob["theta"]
    layers = _resolve_layers(model)
    attn_shape = _get_attention_shape(model, layers)
    lhp, alpha, theta = _filter_compatible_layers(model, tokenizer, attn_shape, lhp, alpha, theta)

    # --- BASE arm (null offsets) ---
    print("\n[arm base] zeroed offsets (identity through the v7hf hook path)", flush=True)
    h = install_lofit_hooks(layers, torch.zeros_like(alpha), torch.zeros_like(theta), lhp, attn_shape)
    try:
        base = score_arm(model, tokenizer, items, args.task, args.apply_chat_template, "base")
    finally:
        remove_handles(h)

    # --- MC arm (trained offsets) ---
    print("\n[arm mc] trained offsets installed", flush=True)
    h = install_lofit_hooks(layers, alpha, theta, lhp, attn_shape)
    try:
        mc = score_arm(model, tokenizer, items, args.task, args.apply_chat_template, "mc")
    finally:
        remove_handles(h)

    # --- SANITY GATE: my conditional naive lift must reproduce ~+0.35 / +0.33 ---
    base_cond_acc = acc_norm_from(base, items, "cond")
    mc_cond_acc = acc_norm_from(mc, items, "cond")
    base_ao_acc = acc_norm_from(base, items, "ao")
    mc_ao_acc = acc_norm_from(mc, items, "ao")
    cond_lift = mc_cond_acc - base_cond_acc
    print("\n" + "=" * 70, flush=True)
    print(f"  SANITY (conditional, acc_norm): base={base_cond_acc:.4f} mc={mc_cond_acc:.4f}"
          f"  lift={cond_lift:+.4f}  (expect ~+0.35 ARC / ~+0.33 TQA)", flush=True)
    if abs(cond_lift) < 0.20:
        print(f"  !!! WARNING: conditional lift {cond_lift:+.4f} far from canonical — scaffold "
              f"likely mismatched (chat-template/continuation). Inspect before trusting answer-only.",
              flush=True)
    else:
        print("  scaffold OK (conditional lift in range) -> answer-only term is trustworthy", flush=True)
    print(f"  CHOICES-ONLY (acc_norm): base={base_ao_acc:.4f} mc={mc_ao_acc:.4f}"
          f"  lift={mc_ao_acc - base_ao_acc:+.4f}  (chance~{np.mean([1/len(o) for _,_,o,_ in items]):.3f})",
          flush=True)
    print("=" * 70, flush=True)

    # --- write per-task file (cond + answer-only, both arms), combine-ready ---
    payload = {
        "task": args.task,
        "neutral_premise": NEUTRAL_PREMISE[args.task],
        "cond_template": COND_PREMISE[args.task],
        "apply_chat_template": args.apply_chat_template,
        "sanity": {"base_cond": base_cond_acc, "mc_cond": mc_cond_acc, "cond_lift": cond_lift,
                   "base_choices_only": base_ao_acc, "mc_choices_only": mc_ao_acc},
        # self-sufficient for scripts/score_mc_pmi.py --premise-file (same-scaffold PMI):
        "gold": {str(did): gold for did, q, opts, gold in items},
        "charlen": {str(did): [max(1, len(o)) for o in opts] for did, q, opts, gold in items},
        "answer_only": {"base": {str(did): base[did]["ao"] for did, *_ in items},
                        "mc": {str(did): mc[did]["ao"] for did, *_ in items}},
        "conditional": {"base": {str(did): base[did]["cond"] for did, *_ in items},
                        "mc": {str(did): mc[did]["cond"] for did, *_ in items}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {args.out}", flush=True)
    print(f"  next: merge per-task files -> answer_only_logprobs.json, then local "
          f"scripts/score_mc_pmi.py", flush=True)


if __name__ == "__main__":
    main()
