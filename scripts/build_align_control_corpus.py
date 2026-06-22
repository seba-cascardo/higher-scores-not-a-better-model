"""Build the matched-data SFT corpus for the Align-LoRA mechanism control.

The P0 reviewer-threat control (arXiv:2508.05078): is the V7-mc lift produced by
the CONTRASTIVE objective, or would ANY fine-tuning on the same data produce it?

To isolate the objective, this control trains a standard SFT LoRA on the EXACT
SAME data and input form as V7-mc, differing ONLY in the objective:
  - V7-mc:  contrastive head-offsets that learn to PREFER `correct` over `wrong`.
  - control: SFT next-token CE that learns to GENERATE `correct` given `question`.

Both see identical (question -> correct) signal on identical splits, so any lift
difference is attributable to the objective, not the data or the input form.

We import build_tqa_pairs / build_arc_pairs from train_v7_lofit so the split is
byte-identical to V7-mc's training data (TQA: seed-0 perm, first 80% of validation;
ARC: native ARC-Challenge train, disjoint from test). NO reimplementation.

Output: JSONL of {task, prompt, target} consumable by train_align_lora.py via
--train-corpus.

Usage (local or pod prelude; deterministic, CPU + HF download only):
    python scripts/build_align_control_corpus.py \
        --out runs/align_lora_control/corpus_matched.jsonl \
        --n-per-task 100000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the CANONICAL split builders so the control's training data is
# byte-identical to V7-mc's. These pull torch + datasets at import time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_v7_lofit import build_tqa_pairs, build_arc_pairs  # noqa: E402


def _pairs_to_sft(pairs, task):
    """(question, correct, wrong) -> {task, prompt=question, target=correct}.

    We drop `wrong` (the SFT control only learns to GENERATE correct; the
    contrastive signal lives only in V7-mc). target is the raw correct choice
    text; train_align_lora.py wraps prompt/target in the chat template.
    """
    out = []
    for q, correct, _wrong in pairs:
        target = str(correct).strip()
        if not q or not target:
            continue
        out.append({"task": task, "prompt": str(q).strip(), "target": target})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    ap.add_argument("--n-per-task", type=int, default=100000,
                    help="Cap per task (default huge = take the full matched split)")
    ap.add_argument("--seed", type=int, default=0, help="MUST be 0 to match V7-mc")
    args = ap.parse_args()

    if args.seed != 0:
        print(f"WARNING: seed={args.seed} != 0 does NOT match V7-mc's training "
              f"split. The control is only clean at seed=0.", flush=True)

    print(f"=== Building matched Align-LoRA control corpus (seed={args.seed}) ===",
          flush=True)
    print("  [identity guarantee] importing build_tqa_pairs / build_arc_pairs "
          "from train_v7_lofit -> same splits as V7-mc", flush=True)

    tqa = build_tqa_pairs(n=args.n_per_task, split="train", seed=args.seed)
    arc = build_arc_pairs(n=args.n_per_task, split="train", seed=args.seed)

    examples = _pairs_to_sft(tqa, "tqa") + _pairs_to_sft(arc, "arc")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_tqa = sum(1 for e in examples if e["task"] == "tqa")
    n_arc = sum(1 for e in examples if e["task"] == "arc")
    print(f"\n=== Corpus written: {args.out} ===", flush=True)
    print(f"  total={len(examples)}  (tqa={n_tqa}, arc={n_arc})", flush=True)
    print("  NOTE: this is V7-mc's EXACT training data (TQA-train-80% seed-0 + "
          "ARC-Challenge-train); the control's TQA eval MUST be the held-out 20% "
          "(see pre-reg R5/L4).", flush=True)
    print("\nNext: python scripts/train_align_lora.py --base <BASE> "
          "--train-corpus " + str(args.out) + " --lora-r 256 --epochs 2 "
          "--out runs/align_lora_control/r256/", flush=True)


if __name__ == "__main__":
    main()
