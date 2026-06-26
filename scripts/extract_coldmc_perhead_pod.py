#!/usr/bin/env python
"""P1 — activation-probe in-space: extract per-head o_proj-input on the cold-MC
option-continuation distribution (base, NO adapter). POD / GPU.

WHY (handoff 2026-06-26 §1d; the prose-freeze gate of §5)
---------------------------------------------------------
§5.2b shows the +35 is NOT a re-scoring of the collapsed output scalars: the
surface stack tops out at AUC 0.574 while the adapter reaches 0.830 (base-wrong
ARC, n=204). But that gap is information-set-confounded: the adapter mutates
ACTIVATIONS, the stack re-weights collapsed SCALARS. This extract feeds the
de-confounded test — fit a linear probe on the SAME per-head o_proj-input
activations the adapter writes to, base/no-adapter, and ask whether a same-space
reader recovers the 0.830:

  ~0.83  -> the correctness signal is present and ANY same-space reader extracts
            it -> pure miscalibration (readout-claim in its strongest form).
  ~0.52  -> the offset adds/rotates signal a linear same-space probe cannot read
            -> representational edit, a stronger-and-different claim.

W_know (LDA, 6.4%) is the on-axis LOWER bound; this probe is the in-space UPPER
bound that is missing.

FORWARD-FILTER (mandatory; [[feedback_forward_filter_same_distribution_same_space]])
------------------------------------------------------------------------------------
  * same-distribution: the cold-MC option-continuation distribution that produced
    the +35 — NOT postgen self-eval, NOT 0-shot. We reconstruct each option prompt
    BYTE-FOR-BYTE from d_null.json `arguments` (the exact (context, continuation)
    lm-eval scored), so there is zero template drift.
  * same-space: per-head o_proj-INPUT (the point LoFiT offsets act on), captured at
    the LAST token of the option-continuation, subset to the adapter's own
    (layer, head) cells — NOT the residual d_model stream.

CRITICAL TOKENIZATION NOTE
--------------------------
The `arguments` strings already contain a literal "<bos>" + the rendered chat
scaffold (lm-eval applied the template before logging). We therefore tokenize with
add_special_tokens=False so the tokenizer does NOT prepend a SECOND bos — a double
bos would be a different distribution. We dump the first prompt's leading token ids
and assert exactly one leading bos to fail loud if this ever drifts.

INPUT  : runs/vinf_causal/d_null.json   (base cold-MC, lm-eval --log-samples)
         runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt  (the 48 adapter (layer,head))
OUTPUT : runs/vinf_causal/coldmc_perhead_<task>.pt  (acts + per-(item,option) meta)
         -> consumed LOCALLY by scripts/analyze_activation_probe_inspace.py

Usage (pod, GPU; cd /workspace/MSAP per block):
  export BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' | grep -v /.locks/ | head -1)")
  python scripts/extract_coldmc_perhead_pod.py --base "$BASE" \
      --null runs/vinf_causal/d_null.json --task arc_challenge \
      --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --out runs/vinf_causal/coldmc_perhead_arc_challenge.pt
Compile-check:  python -c "import ast;ast.parse(open('scripts/extract_coldmc_perhead_pod.py').read())"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the canonical V7 per-head capture machinery (byte-identical shape
# detection + o_proj pre-hook) so this lives on the SAME grid as v_inf / theta_mc.
from scripts.probe_v7_lofit_head_selection import (
    _resolve_layers,
    _resolve_text_config,
    _get_attention_shape_runtime,
    attach_per_head_capture,
    remove_handles,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCHEMA_VERSION = "coldmc-perhead-1.0"


# ---------------------------------------------------------------------------------
# d_null.json loaders (mirror analyze_cold_mc_calibration.py)
# ---------------------------------------------------------------------------------
def load_task_samples(path: Path, task: str) -> list[dict]:
    d = json.load(path.open(encoding="utf-8"))
    if task not in d["samples"]:
        raise KeyError(f"{path.name}: task '{task}' missing. Have: "
                       f"{list(d['samples'].keys())[:6]}...")
    return d["samples"][task]


def option_texts(doc: dict) -> list[str]:
    ch = doc.get("choices")
    if isinstance(ch, dict) and "text" in ch:
        return list(ch["text"])
    if isinstance(ch, list) and ch and isinstance(ch[0], str):
        return list(ch)
    mc1 = doc.get("mc1_targets")
    if isinstance(mc1, dict) and "choices" in mc1:
        return list(mc1["choices"])
    return []


def option_prompts_from_arguments(sample: dict) -> list[str]:
    """Reconstruct the EXACT per-option string lm-eval scored: context+continuation.

    `arguments` is a list of [context, continuation] pairs, one per option, with the
    chat template already rendered (incl. a literal <bos>). The capture position is
    the LAST token of (context+continuation) = the last token of the option.
    """
    args = sample["arguments"]
    out = []
    for pair in args:
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            raise ValueError(f"unexpected arguments shape for doc_id={sample.get('doc_id')}: {pair!r:.120}")
        out.append(pair[0] + pair[1])
    return out


def cond_logprobs(sample: dict) -> np.ndarray:
    return np.array([float(r[0]) for r in sample["filtered_resps"]], dtype=np.float64)


def base_acc_norm_pick(sample: dict) -> tuple[int, int]:
    """(base_pick, base_correct) under acc_norm (char-length-normalised), the +35 metric."""
    texts = option_texts(sample["doc"])
    raw = cond_logprobs(sample)
    if not texts or len(texts) != len(raw):
        # fall back to raw logprob argmax if option texts unavailable
        pick = int(np.argmax(raw))
    else:
        clen = np.array([max(1, len(t)) for t in texts], dtype=np.float64)
        pick = int(np.argmax(raw / clen))
    gold = int(sample["target"])
    return pick, int(pick == gold)


# ---------------------------------------------------------------------------------
# per-head capture on a list of pre-rendered prompts (add_special_tokens=False)
# ---------------------------------------------------------------------------------
def capture_perhead_no_special(model, tokenizer, prompts, device, attn_shape,
                               n_layers, seq_len_max, layer_dims, bos_id):
    """Like capture_per_head_activations but tokenizes with add_special_tokens=False.

    The prompts already carry the rendered <bos>+scaffold (from lm-eval `arguments`);
    auto-prepending another bos would change the distribution. Returns
    (tensor[n_prompts, n_captured_layers, num_q_heads, head_dim], captured_indices).
    """
    layers = _resolve_layers(model)
    captured: dict = {}
    handles = attach_per_head_capture(layers, attn_shape, captured, layer_dims)
    if layer_dims is None:
        captured_indices = list(range(n_layers))
    else:
        captured_indices = sorted(
            li for li in range(n_layers) if layer_dims.get(li) == attn_shape.total_q_dim
        )
    out = []
    t0 = time.time()
    try:
        for i, prompt in enumerate(prompts):
            captured.clear()
            toks = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=seq_len_max, add_special_tokens=False).to(device)
            if i == 0:
                lead = toks["input_ids"][0, :4].tolist()
                print(f"  [tok-sanity] first prompt leading ids={lead} (bos_id={bos_id}); "
                      f"len={toks['input_ids'].shape[1]}", flush=True)
                if bos_id is not None:
                    assert lead and lead[0] == bos_id, (
                        f"first token {lead[0]} != bos {bos_id}: the <bos> in the rendered "
                        f"string was not mapped — check split_special_tokens / template.")
                    assert len(lead) < 2 or lead[1] != bos_id, (
                        f"double bos {lead[:2]} — add_special_tokens leaked a second bos.")
            with torch.no_grad():
                model(**toks, use_cache=False)
            per_layer = []
            for li in captured_indices:
                if li not in captured:
                    raise RuntimeError(f"Layer {li} not captured (expected match).")
                per_layer.append(captured[li])
            out.append(torch.stack(per_layer, dim=0))
            if (i + 1) % 200 == 0:
                print(f"      forward {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        remove_handles(handles)
    return torch.stack(out, dim=0), captured_indices


# ---------------------------------------------------------------------------------
# adapter head alignment (mirror extract_functional_axis_perhead._load_theta_mc)
# ---------------------------------------------------------------------------------
def adapter_cells(offsets_path: Path, captured_indices, num_q_heads, head_dim):
    """Return [(arr_idx, head, layer)] for the adapter's (layer,head) cells that
    align to the captured grid — the same-space subset for the probe."""
    blob = torch.load(offsets_path, weights_only=False, map_location="cpu")
    pairs = [tuple(int(x) for x in pr) for pr in blob["layer_head_pairs"]]
    theta_hd = int(blob.get("head_dim", blob["theta"].shape[-1])) if "theta" in blob else head_dim
    layer_to_arr = {li: ai for ai, li in enumerate(captured_indices)}
    cells, skipped_layer, skipped_hd = [], 0, 0
    for (li, hi) in pairs:
        if li not in layer_to_arr:
            skipped_layer += 1
            continue
        if theta_hd != head_dim:
            skipped_hd += 1
            continue
        if 0 <= hi < num_q_heads:
            cells.append((layer_to_arr[li], hi, li))
    print(f"  adapter offsets: {len(pairs)} pairs -> aligned {len(cells)} to captured grid "
          f"(skipped {skipped_layer} layer-not-captured, {skipped_hd} head_dim-mismatch).", flush=True)
    return cells, {"n_pairs": len(pairs), "skipped_layer": skipped_layer,
                   "skipped_headdim": skipped_hd, "theta_head_dim": theta_hd}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="HF id or local path to Gemma 4 31B IT")
    ap.add_argument("--null", type=Path, default=Path("runs/vinf_causal/d_null.json"),
                    help="lm-eval --log-samples JSON of the BASE cold-MC run")
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--offsets", type=Path,
                    default=Path("runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt"))
    ap.add_argument("--seq-len-max", type=int, default=1024)
    ap.add_argument("--all-heads", action="store_true",
                    help="ALSO store every captured head (large) for an optional looser "
                         "upper bound. Default: store only the adapter's same-space cells.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all items in the task")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or (args.null.parent / f"coldmc_perhead_{args.task}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"P1 EXTRACT — per-head o_proj-input cold-MC | task={args.task}", flush=True)
    print("=" * 70, flush=True)

    samples = load_task_samples(args.null, args.task)
    if args.limit:
        samples = samples[:args.limit]
    print(f"  items: {len(samples)}  from {args.null.name}", flush=True)

    # -- Phase 1: flatten (item, option) -> prompts, keep the join map ----------
    prompts, prompt_meta, items = [], [], []
    for it_idx, s in enumerate(samples):
        did = int(s["doc_id"]); gold = int(s["target"])
        opts = option_prompts_from_arguments(s)
        n_opt = len(opts)
        b_pick, b_corr = base_acc_norm_pick(s)
        pidx0 = len(prompts)
        for o, p in enumerate(opts):
            prompts.append(p)
            prompt_meta.append({"item_idx": it_idx, "doc_id": did, "opt": o,
                                "is_gold": int(o == gold)})
        items.append({"item_idx": it_idx, "doc_id": did, "target": gold, "n_opt": n_opt,
                      "base_pick": b_pick, "base_correct": b_corr,
                      "prompt_idxs": list(range(pidx0, pidx0 + n_opt))})
    n_bw = sum(1 for it in items if it["base_correct"] == 0)
    print(f"  flattened {len(prompts)} (item,option) prompts | base-wrong items: {n_bw}/{len(items)}",
          flush=True)
    print(f"  sample option prompt[-1] tail[:160]: {prompts[items[0]['prompt_idxs'][-1]][-160:]!r}",
          flush=True)

    # -- Phase 2: model + attention-shape detection ----------------------------
    print("Phase 2: loading model + runtime attention-shape detection", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = _resolve_layers(model)
    n_layers = len(layers)
    attn_shape, layer_dims = _get_attention_shape_runtime(model, tokenizer, layers)
    num_q_heads, head_dim = attn_shape.num_q_heads, attn_shape.head_dim
    device = next(model.parameters()).device
    bos_id = getattr(tokenizer, "bos_token_id", None)
    print(f"  layers={n_layers} num_q_heads={num_q_heads} head_dim={head_dim} "
          f"total_q_dim={attn_shape.total_q_dim} bos_id={bos_id}", flush=True)

    # -- Phase 3: capture (last token, o_proj pre-hook, NO extra bos) -----------
    print("Phase 3: capturing per-head o_proj-input (last token)", flush=True)
    H, captured_indices = capture_perhead_no_special(
        model, tokenizer, prompts, str(device), attn_shape, n_layers,
        args.seq_len_max, layer_dims, bos_id)
    H = H.to(torch.float32)
    print(f"  H {tuple(H.shape)}  captured layers ({len(captured_indices)}): {captured_indices}",
          flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Phase 4: subset to the adapter's same-space cells ---------------------
    cells, align_info = adapter_cells(args.offsets, captured_indices, num_q_heads, head_dim)
    if not cells:
        raise RuntimeError("0 adapter cells aligned to the captured grid — same-space subset "
                           "is empty; cannot build the in-space probe. Check offsets/grid.")
    arr_idx = torch.tensor([c[0] for c in cells], dtype=torch.long)
    head_idx = torch.tensor([c[1] for c in cells], dtype=torch.long)
    # acts_adapter: (n_prompts, n_cells, head_dim)
    acts_adapter = H[:, arr_idx, head_idx, :].contiguous()
    print(f"  acts_adapter {tuple(acts_adapter.shape)} (n_prompts, n_adapter_heads, head_dim)",
          flush=True)

    state = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "base": args.base,
        "n_items": len(items),
        "n_base_wrong": n_bw,
        "num_q_heads": num_q_heads,
        "head_dim": head_dim,
        "captured_indices": captured_indices,
        "adapter_cells_layer_head": [(c[2], c[1]) for c in cells],  # (layer, head)
        "adapter_align_info": align_info,
        "prompt_meta": prompt_meta,
        "items": items,
        "acts_adapter": acts_adapter,            # (n_prompts, n_adapter_heads, head_dim)
        "forward_filter": {
            "same_distribution": "cold-MC option-continuation reconstructed from d_null arguments",
            "same_space": "per-head o_proj-input at last token, adapter (layer,head) subset",
        },
    }
    if args.all_heads:
        state["acts_all"] = H                    # (n_prompts, n_captured_layers, num_q_heads, head_dim)
        print(f"  [--all-heads] also storing acts_all {tuple(H.shape)} (large)", flush=True)

    torch.save(state, out_path)
    sz = out_path.stat().st_size / 1e6
    print(f"\n  saved: {out_path}  ({sz:.0f} MB)", flush=True)
    print("  NEXT (local): python scripts/analyze_activation_probe_inspace.py "
          f"--acts {out_path}", flush=True)


if __name__ == "__main__":
    main()
