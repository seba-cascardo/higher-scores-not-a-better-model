"""W_know-of-Align / W_know-of-contrastive (clean, apples-to-apples): on-axis or off-axis?

Captures last-token per-head o_proj-input over ARC for base vs a TREATED model, then
cos(diff, v_inf) per head. The treated model is either:
  --align <peft_dir>   : a PEFT Align-LoRA (merged), OR
  --offsets <lofit.pt> : the LoFiT contrastive adapter (hooks applied; capture sees the
                         offset-modified o_proj input).
Both are measured as the NET EFFECT on activations, so Align vs contrastive is a fair
head-to-head (the raw theta_mc offset cos 0.0014 is NOT the same object as the net effect).

  ratio |cos(diff,v_inf)|/|cos(diff,random)| ~1 -> off-axis ; >>1 -> on-axis.

Reuses the canonical per-head capture (probe_v7_lofit_head_selection) + LoFiT hook install
(eval_with_router) + compat filter (run_lm_eval_v7). GPU (Gemma 31B). Pod.

Usage (pod):
  # Align (PEFT):
  python scripts/extract_align_direction.py --base "$BASE" --align runs/align_lora_control/r256 \
      --vinf runs/derisk/mc_vinf_offset.pt --n 200 --out runs/derisk/align_direction.json
  # Contrastive (LoFiT, apples-to-apples):
  python scripts/extract_align_direction.py --base "$BASE" \
      --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --vinf runs/derisk/mc_vinf_offset.pt --n 200 --out runs/derisk/contrastive_direction.json
"""
import sys
import json
import argparse
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.probe_v7_lofit_head_selection import (  # noqa: E402
    capture_per_head_activations,
    _get_attention_shape_runtime,
    _resolve_layers,
)


def build_arc_prompts(tok, n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    out = []
    for i in idx:
        ex = ds[i]
        opts = "\n".join(f"({l}) {t}" for t, l in
                         zip(ex["choices"]["text"], ex["choices"]["label"]))
        user = f"Question: {ex['question']}\nChoices:\n{opts}\nAnswer:"
        out.append(tok.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True))
    return out


def capture(model, tok, prompts, device, offsets_path=None):
    """Per-head o_proj-input capture (last token). If offsets_path is given, LoFiT hooks
    are installed FIRST so the capture pre-hook (registered after) sees the offset-modified
    input — the contrastive adapter's NET effect, comparable to the Align's diff."""
    layers = _resolve_layers(model)
    attn, layer_dims = _get_attention_shape_runtime(model, tok, layers)
    lofit_h = []
    if offsets_path:
        from scripts.eval_with_router import install_lofit_hooks
        from scripts.run_lm_eval_v7 import _filter_compatible_layers
        blob = torch.load(offsets_path, weights_only=False)
        lhp, alpha, theta = blob["layer_head_pairs"], blob["alpha"], blob["theta"]
        lhp, alpha, theta = _filter_compatible_layers(model, tok, attn, lhp, alpha, theta)
        lofit_h = install_lofit_hooks(layers, alpha, theta, lhp, attn)
    try:
        acts, cap_idx = capture_per_head_activations(
            model, tok, prompts, device, attn, len(layers), 2048, layer_dims)
    finally:
        if lofit_h:
            from scripts.eval_with_router import remove_handles
            remove_handles(lofit_h)
    return acts, cap_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--align", help="PEFT adapter dir (e.g. r256)")
    ap.add_argument("--offsets", help="LoFiT contrastive offsets .pt (alternative to --align)")
    ap.add_argument("--vinf", required=True, help="mc_vinf_offset.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if bool(args.align) == bool(args.offsets):
        ap.error("provide exactly one of --align / --offsets")
    device = "cuda"
    label = "align" if args.align else "contrastive"

    tok = AutoTokenizer.from_pretrained(args.base)
    prompts = build_arc_prompts(tok, args.n)
    print(f"[{label}-dir] {len(prompts)} ARC-test prompts (chat-template, last token)", flush=True)

    vinf = torch.load(args.vinf, weights_only=False)
    vpairs = [tuple(int(x) for x in p) for p in vinf["layer_head_pairs"]]
    vtheta = vinf["theta"].float()

    print("[base] load + capture ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=device).eval()
    base_acts, cap_idx = capture(base, tok, prompts, device)

    if args.offsets:
        print("[contrastive] capture WITH LoFiT offsets (same model) ...", flush=True)
        treated_acts, cap_idx2 = capture(base, tok, prompts, device, offsets_path=args.offsets)
    else:
        del base
        torch.cuda.empty_cache()
        print("[align] load + merge_and_unload + capture ...", flush=True)
        from peft import PeftModel
        b2 = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map=device).eval()
        treated = PeftModel.from_pretrained(b2, args.align).merge_and_unload().eval()
        treated_acts, cap_idx2 = capture(treated, tok, prompts, device)

    assert cap_idx == cap_idx2, "captured layers differ base vs treated"
    cap_map = {li: k for k, li in enumerate(cap_idx)}

    rows = []
    for i, (L, h) in enumerate(vpairs):
        if L not in cap_map:
            continue
        k = cap_map[L]
        diff = (treated_acts[:, k, h, :].float() - base_acts[:, k, h, :].float()).mean(0)
        dn = diff.norm()
        v = vtheta[i]
        cos = float(torch.dot(diff, v) / (dn * v.norm() + 1e-9))
        g = torch.Generator().manual_seed(1000 + i)
        rc = []
        for _ in range(50):
            r = torch.randn(diff.numel(), generator=g)
            rc.append(abs(float(torch.dot(diff, r) / (dn * r.norm() + 1e-9))))
        rows.append({"L": L, "h": h, "cos_vinf": cos, "abs_cos_vinf": abs(cos),
                     "mean_abs_cos_rand": statistics.mean(rc), "diff_norm": float(dn)})

    ac = [r["abs_cos_vinf"] for r in rows]
    rcm = [r["mean_abs_cos_rand"] for r in rows]
    ratio = statistics.mean(ac) / max(statistics.mean(rcm), 1e-9)
    dnorms = [r["diff_norm"] for r in rows]
    print(f"\n[{label}-dir] heads matched: {len(rows)}/{len(vpairs)}")
    print(f"[{label}-dir] diff_norm: min {min(dnorms):.2f} max {max(dnorms):.2f} mean {statistics.mean(dnorms):.2f}  "
          f"(contrastive sanity: ~offset norms 3.5-17.7 => offsets applied)")
    print(f"[{label}-dir] |cos(diff, v_inf)| : mean {statistics.mean(ac):.4f}  median {statistics.median(ac):.4f}")
    print(f"[{label}-dir] |cos(diff, random)|: mean {statistics.mean(rcm):.4f}  (256-dim baseline)")
    print(f"[{label}-dir] ratio v_inf/random = {ratio:.2f}   (Align measured 3.93)")
    print(f"[{label}-dir] VERDICT: ratio ~1 -> OFF-AXIS net effect ; >>1 -> ON-AXIS net effect")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": {"label": label, "mean_abs_cos_vinf": statistics.mean(ac),
                           "mean_abs_cos_rand": statistics.mean(rcm), "ratio": ratio,
                           "n_heads": len(rows)}, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"[{label}-dir] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
