"""W_know-of-Align (clean): is the Align-LoRA's EFFECTIVE direction on-axis or off-axis?

Captures per-head o_proj-input activations (LAST token) over ARC with base vs Align
(merged), computes diff = mean_prompts(align - base) per head = the Align's effective
direction in the SAME per-head space where v_inf lives, then cos(diff, v_inf).

  |cos(diff, v_inf)| ~ random baseline  -> OFF-AXIS  (confirms the weight-space preview
                                           ratio 1.05 + theta_mc cos 0.0014; the generic
                                           component is off-axis, the thesis generalizes)
  |cos(diff, v_inf)| >> random          -> ON-AXIS   (the off-axis claim is method-specific)

Reuses the canonical per-head capture (last-token o_proj input, Gemma-4 hetero-safe) from
probe_v7_lofit_head_selection. GPU (Gemma 31B x2, sequential). Pod.

Usage (pod):
  python scripts/extract_align_direction.py --base "$BASE" \
      --align runs/align_lora_control/r256 --vinf runs/derisk/mc_vinf_offset.pt \
      --n 200 --out runs/derisk/align_direction.json
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


def capture(model, tok, prompts, device):
    layers = _resolve_layers(model)
    attn, layer_dims = _get_attention_shape_runtime(model, tok, layers)
    acts, cap_idx = capture_per_head_activations(
        model, tok, prompts, device, attn, len(layers), 2048, layer_dims)
    return acts, cap_idx  # (n_prompts, n_cap_layers, num_q_heads, head_dim), list[int]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--align", required=True, help="PEFT adapter dir (e.g. r256)")
    ap.add_argument("--vinf", required=True, help="mc_vinf_offset.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base)
    prompts = build_arc_prompts(tok, args.n)
    print(f"[align-dir] {len(prompts)} ARC-test prompts (chat-template, last token)", flush=True)

    vinf = torch.load(args.vinf, weights_only=False)
    vpairs = [tuple(int(x) for x in p) for p in vinf["layer_head_pairs"]]
    vtheta = vinf["theta"].float()

    print("[base] load + capture ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=device).eval()
    base_acts, cap_idx = capture(base, tok, prompts, device)
    del base
    torch.cuda.empty_cache()

    print("[align] load + merge_and_unload + capture ...", flush=True)
    from peft import PeftModel
    b2 = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=device).eval()
    align = PeftModel.from_pretrained(b2, args.align).merge_and_unload().eval()
    align_acts, cap_idx2 = capture(align, tok, prompts, device)
    assert cap_idx == cap_idx2, "captured layers differ between base and align"
    cap_map = {li: k for k, li in enumerate(cap_idx)}

    rows = []
    for i, (L, h) in enumerate(vpairs):
        if L not in cap_map:
            continue
        k = cap_map[L]
        diff = (align_acts[:, k, h, :].float() - base_acts[:, k, h, :].float()).mean(0)
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
    print(f"\n[align-dir] heads matched: {len(rows)}/{len(vpairs)}")
    print(f"[align-dir] |cos(diff_Align, v_inf)| : mean {statistics.mean(ac):.4f}  median {statistics.median(ac):.4f}")
    print(f"[align-dir] |cos(diff_Align, random)|: mean {statistics.mean(rcm):.4f}  (256-dim baseline)")
    print(f"[align-dir] ratio v_inf/random = {ratio:.2f}")
    print(f"[align-dir] REFERENCE cos(theta_mc, v_inf) = 0.0014 (contrastive, off-axis)")
    print(f"[align-dir] VERDICT: ratio ~1 -> OFF-AXIS (generic component off-axis, thesis generalizes); "
          f"ratio >>1 -> ON-AXIS (off-axis is method-specific)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": {"mean_abs_cos_vinf": statistics.mean(ac),
                           "mean_abs_cos_rand": statistics.mean(rcm),
                           "ratio": ratio, "n_heads": len(rows)}, "rows": rows},
              open(args.out, "w"), indent=1)
    print(f"[align-dir] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
