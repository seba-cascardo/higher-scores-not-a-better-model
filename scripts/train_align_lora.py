"""Train a single high-rank Align-LoRA adapter for Sprint 1.5 A.2 falsification.

This is the **P0 reviewer-threat defensive ablation**: if a single high-rank
LoRA, trained on the union of RoMuLo's 4 task-training corpora (ARC, MMLU,
GSM8K, SocialIQA), matches multi-adapter RoMuLo on in-distribution AND has
comparable spillover, then RoMuLo's multi-adapter premise is killable by a
reviewer and the paper reframes as a Pareto-frontier story.

Reference: arXiv:2508.05078 (verified verbatim 2026-05-22).

# Design

- **Adapter**: PEFT LoraConfig with r=256 (default), targeting all attention
  projections (q/k/v/o). Auto-detects Gemma 4's `Gemma4ClippableLinear.linear`
  wrapper vs standard `nn.Linear` (Qwen/Llama/Mistral).
- **Training data**: union of {ARC Challenge, MMLU, GSM8K, SocialIQA} training
  splits, formatted as SFT (prompt → target). Sampled to `--n-per-task`
  (default 500) for ~2K total examples.
- **Loss**: standard causal LM cross-entropy, prefix-masked to count only
  target tokens (same approach as scripts/train_router_lora.py:RouterDataset).
- **Base**: Gemma 4 31B IT (canonical RoMuLo deploy target). BF16 by default;
  use --load-in-4bit on smaller pod GPUs (QLoRA-style).

# Memory budget

| Config | Model weights | LoRA params | AdamW state | Activations (grad-ckpt, bs=1, seq=1024) | Total est. |
|---|---|---|---|---|---|
| BF16 + r=256, no grad-ckpt | 62 GB | ~540 MB | ~4.3 GB | ~12 GB | ~79 GB |
| BF16 + r=256, grad-ckpt | 62 GB | ~540 MB | ~4.3 GB | ~3 GB | ~70 GB |
| 4-bit + r=256, grad-ckpt | 16 GB | ~540 MB | ~4.3 GB | ~3 GB | ~24 GB |

→ Single 48 GB GPU (RTX PRO 6000 Blackwell): use --load-in-4bit.
→ Single 80 GB GPU (H100/A100-80GB): BF16 + grad-ckpt fits.

# Fairness vs RoMuLo multi-adapter setup

The comparison is "single high-rank LoRA on UNION of training data" vs
"4 specialized adapters, each on its own task's training data". Both see
the SAME total training inputs; the only difference is single-adapter
capacity vs routed specialization. Eval is the SAME 4 tasks via
lm-eval-harness (handled by a separate eval script, not this trainer).

MMLU does NOT appear in any individual RoMuLo adapter's training corpus
(V7-mc=TQA+ARC, V7-comp=Lambada, C.3=math_proof, D.1=SocialIQA), but is
in this Align-LoRA corpus to match the eval-task spec. This is a small
asymmetry; the more important comparison is ARC/GSM8K/SocialIQA where
RoMuLo also trains. MMLU is most useful as a SPILLOVER probe (does
Align-LoRA spillover-help MMLU more than RoMuLo? — Property 2 test).

# Usage (pod, post bring-up + git pull)

    # 1. Single-paste training (loads HF datasets in-script)
    python scripts/train_align_lora.py \\
        --base /workspace/.hf_home/models/gemma-4-31b-it \\
        --n-per-task 500 \\
        --epochs 2 \\
        --lora-r 256 \\
        --batch-size 1 \\
        --grad-accum 8 \\
        --load-in-4bit \\
        --out runs/align_lora/r256_n500_ep2/

    # 2. Or pre-build corpus once, train multiple ranks
    python scripts/train_align_lora.py --build-corpus-only \\
        --n-per-task 500 \\
        --out-corpus runs/align_lora/corpus_n500.jsonl

    python scripts/train_align_lora.py \\
        --base /workspace/.hf_home/models/gemma-4-31b-it \\
        --train-corpus runs/align_lora/corpus_n500.jsonl \\
        --lora-r 256 \\
        --out runs/align_lora/r256_ep2/

# Eval (separate, after training)

    # Run lm-eval-harness against the saved adapter for each of 4 tasks.
    # See scripts/run_lm_eval_v7*.py for the canonical wrapper pattern;
    # swap the adapter arg to the Align-LoRA output dir.

# Decision gate (per handoff 2026-05-23-NEXT-SESSION-START-HERE-sprint-1.5-week1.md)

- Align-LoRA wins in-dist by >5pp AND comparable spillover → paper reframes
  as "Pareto frontier" (multi-adapter for Property 2, single for in-dist).
- Else → A.2 becomes a strong defensive ablation citation, multi-adapter
  framing stays.

# Cross-refs

- Spec: docs/superpowers/specs/2026-05-23-NEXT-SESSION-START-HERE-sprint-1.5-week1.md §2.1
- Roadmap: docs/superpowers/plans/2026-05-20-romulo-roadmap-to-publication.md Sprint 1.5 A.2
- Risk: STATUS.md P0 reviewer risks #4
- Anti-pattern guard for HF dataset loading: memory feedback_hf_dataset_script_deprecation.md
  (datasets>=3 blocks Python loaders; use parquet revisions)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple


# Required deps: pip install peft accelerate datasets bitsandbytes (for 4-bit)
def _check_deps(need_4bit: bool = False):
    missing = []
    for mod in ("peft", "accelerate", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if need_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            missing.append("bitsandbytes")
    if missing:
        print(f"ERROR: missing dependencies. Install with:", file=sys.stderr)
        print(f"  pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)


# Multimodal submodules to skip when detecting text-decoder LoRA targets.
# Gemma 4 loads as a ConditionalGeneration model whose vision/audio towers use a
# Gemma4ClippableLinear wrapper (.linear inner). The text decoder uses plain
# nn.Linear. If detection hits a tower's q_proj first it returns ".linear"
# suffixes that ONLY match the towers -> on a text corpus the towers never run,
# the LoRA gets zero gradient, lora_B stays at its zero-init, and the adapter is
# a complete no-op (A.2 post-mortem 2026-06-02: flat loss, eval == base).
_NON_TEXT_SUBTREES = ("vision_tower", "audio_tower", "vision_model", "audio_model")


def _detect_target_modules(model,
                            candidates=("q_proj", "k_proj", "v_proj", "o_proj")):
    """Auto-detect LoRA target_modules suffix list for the TEXT decoder.

    Skips multimodal vision/audio towers (see `_NON_TEXT_SUBTREES`) so a text-only
    Align-LoRA actually adapts the language model rather than an inert tower.

    Returns:
      - Standard nn.Linear (Qwen, Llama, Mistral, Gemma 4 text): ["q_proj","k_proj",...]
      - Pure Gemma4ClippableLinear wrapper, no towers (.linear inner):
        ["q_proj.linear", "k_proj.linear", ...]
    """
    import torch.nn as nn
    for name, module in model.named_modules():
        if any(t in name for t in _NON_TEXT_SUBTREES):
            continue  # skip vision/audio towers -> target the text decoder only
        if any(name.endswith("." + c) for c in candidates):
            if isinstance(module, nn.Linear):
                return list(candidates)
            if hasattr(module, "linear") and isinstance(module.linear, nn.Linear):
                return [f"{c}.linear" for c in candidates]
            raise ValueError(
                f"Module {name!r} is type {type(module).__name__} (neither nn.Linear "
                f"nor a wrapper with .linear inner). Pass --target-modules manually."
            )
    raise ValueError(
        f"Could not find any text-decoder attention modules matching {candidates} "
        f"(searched outside {_NON_TEXT_SUBTREES}). Pass --target-modules manually."
    )


def load_tokenizer_compat(model_path):
    """Mirror train_router_lora.py — handle Mistral regex fix transparently."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except (TypeError, ValueError):
        return AutoTokenizer.from_pretrained(model_path)


# ---------- Task-specific SFT formatters ----------------------------------
# Format choice rationale: match lm-eval-harness default templates for each
# task so the trained adapter behaves correctly at eval time. Variations
# between {evaluator's} prompt format and {trainer's} prompt format are a
# common silent failure mode (see memory feedback_chat_template_gap).

def _format_arc(row: dict) -> Tuple[str, str]:
    """ARC Challenge / Easy. Standard lm-eval doc-to-text format."""
    q = row["question"]
    choices = row["choices"]
    labels = choices["label"]    # ['A','B','C','D'] typically
    texts = choices["text"]
    answer_key = row["answerKey"]  # e.g., 'B'
    if answer_key.isdigit():
        # Some ARC rows label numerically (1/2/3/4); align to letters
        idx = int(answer_key) - 1
        answer_key = labels[idx]
    choice_lines = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, texts))
    prompt = f"Question: {q}\n{choice_lines}\nAnswer:"
    target = f" {answer_key}"
    return prompt, target


def _format_mmlu(row: dict) -> Tuple[str, str]:
    """MMLU (cais/mmlu). 4-choice MC with `answer` int 0-3."""
    q = row["question"]
    choices = row["choices"]  # list[str] of length 4
    answer_idx = int(row["answer"])  # 0-3
    labels = ["A", "B", "C", "D"]
    choice_lines = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, choices))
    prompt = f"Question: {q}\n{choice_lines}\nAnswer:"
    target = f" {labels[answer_idx]}"
    return prompt, target


def _format_gsm8k(row: dict) -> Tuple[str, str]:
    """GSM8K main split. Train on full chain-of-thought + boxed answer."""
    q = row["question"]
    a = row["answer"]   # already contains '#### <final>' marker
    prompt = f"Question: {q}\nAnswer:"
    target = f" {a}"
    return prompt, target


def _format_socialiqa(row: dict) -> Tuple[str, str]:
    """SocialIQA (lighteval/siqa parquet). MC with context + question + 3 options.

    Per memory feedback_hf_dataset_script_deprecation: datasets>=3 blocks the
    legacy Python loader for allenai/social_i_qa; lighteval/siqa is the
    parquet-mirrored alternative used elsewhere in the repo.
    """
    ctx = row["context"]
    q = row["question"]
    a, b, c = row["answerA"], row["answerB"], row["answerC"]
    label = row["label"]  # '1' / '2' / '3' as string per siqa convention
    labels_map = {"1": "A", "2": "B", "3": "C"}
    answer_letter = labels_map.get(str(label).strip(), str(label))
    prompt = (
        f"Context: {ctx}\nQuestion: {q}\n"
        f"A. {a}\nB. {b}\nC. {c}\nAnswer:"
    )
    target = f" {answer_letter}"
    return prompt, target


# ---------- HF dataset loaders --------------------------------------------

def _load_arc(n: int, seed: int) -> List[dict]:
    """Load ARC Challenge train split."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    ds = ds.shuffle(seed=seed)
    return list(ds.select(range(min(n, len(ds)))))


def _load_mmlu(n: int, seed: int) -> List[dict]:
    """Load MMLU. Uses 'auxiliary_train' (largest train-time pool, ~100K) and
    samples uniformly across subjects via shuffle. For deterministic per-subject
    coverage, replace with a stratified sampler. For bake-off we want broad
    topic coverage so uniform shuffle is the simpler default.
    """
    from datasets import load_dataset
    # cais/mmlu auxiliary_train is the canonical training pool (MMLU eval is on test/validation)
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    ds = ds.shuffle(seed=seed)
    return list(ds.select(range(min(n, len(ds)))))


def _load_gsm8k(n: int, seed: int) -> List[dict]:
    """Load GSM8K main train split."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed)
    return list(ds.select(range(min(n, len(ds)))))


def _load_socialiqa(n: int, seed: int) -> List[dict]:
    """Load SocialIQA train split via lighteval parquet mirror.

    Falls back to allenai/social_i_qa parquet revision if lighteval not
    available — both should yield the same schema {context, question,
    answerA/B/C, label}.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("lighteval/siqa", split="train")
    except Exception:
        ds = load_dataset("allenai/social_i_qa", split="train")
    ds = ds.shuffle(seed=seed)
    return list(ds.select(range(min(n, len(ds)))))


TASK_REGISTRY: dict[str, Tuple[Callable, Callable]] = {
    "arc":       (_load_arc,       _format_arc),
    "mmlu":      (_load_mmlu,      _format_mmlu),
    "gsm8k":     (_load_gsm8k,     _format_gsm8k),
    "socialiqa": (_load_socialiqa, _format_socialiqa),
}


def build_corpus_from_hf(tasks: List[str], n_per_task: int, seed: int,
                          out_path: Path | None = None) -> List[dict]:
    """Load + format all task corpora. Returns flat list of
    {task, prompt, target} dicts. Optionally writes JSONL."""
    corpus: List[dict] = []
    rng = random.Random(seed)
    for task in tasks:
        if task not in TASK_REGISTRY:
            raise ValueError(f"Unknown task {task!r}; known: {sorted(TASK_REGISTRY.keys())}")
        loader, formatter = TASK_REGISTRY[task]
        print(f"  loading {task} (n={n_per_task})...", flush=True)
        t0 = time.time()
        rows = loader(n_per_task, seed)
        formatted = 0
        skipped = 0
        for r in rows:
            try:
                prompt, target = formatter(r)
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    WARN: {task} row format failed ({type(e).__name__}: {e}); skipping",
                          flush=True)
                continue
            corpus.append({"task": task, "prompt": prompt, "target": target})
            formatted += 1
        print(f"    {task}: {formatted} examples (skipped {skipped}, {time.time()-t0:.1f}s)",
              flush=True)
    rng.shuffle(corpus)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for ex in corpus:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  corpus written: {out_path} ({len(corpus)} rows)", flush=True)
    return corpus


def load_corpus_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------- Training dataset + collate ------------------------------------

def format_example_for_training(prompt: str, target: str, tokenizer) -> Tuple[str, str]:
    """Apply chat template + return (input_part, target_part) for loss masking.

    Mirrors scripts/train_router_lora.py:format_example. Critical for matching
    the IT model's instruction-tuning regime — wrong template = LoRA effectively
    untrained at eval time (silent failure mode).
    """
    messages_input = [{"role": "user", "content": prompt}]
    messages_full = messages_input + [{"role": "assistant", "content": target}]
    input_part = tokenizer.apply_chat_template(
        messages_input, tokenize=False, add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False,
    )
    if full.startswith(input_part):
        target_part = full[len(input_part):]
    else:
        # Defensive fallback when add_generation_prompt on/off renders differ
        eos = tokenizer.eos_token or ""
        target_part = target + eos
    return input_part, target_part


class AlignLoRADataset:
    def __init__(self, examples: List[dict], tokenizer, max_length: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        import torch
        ex = self.examples[idx]
        input_part, target_part = format_example_for_training(
            ex["prompt"], ex["target"], self.tokenizer,
        )
        full = input_part + target_part

        enc = self.tokenizer(
            full, truncation=True, max_length=self.max_length,
            return_tensors="pt", add_special_tokens=False,
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        prefix_enc = self.tokenizer(input_part, return_tensors="pt", add_special_tokens=False)
        prefix_len = prefix_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prefix_len] = -100   # mask prefix in loss

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch, pad_token_id: int):
    """Pad to longest sequence in batch."""
    import torch
    max_len = max(b["input_ids"].size(0) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        n = b["input_ids"].size(0)
        pad = max_len - n
        out["input_ids"].append(torch.cat([b["input_ids"],
                                           torch.full((pad,), pad_token_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([b["attention_mask"],
                                                torch.zeros(pad, dtype=torch.long)]))
        out["labels"].append(torch.cat([b["labels"],
                                        torch.full((pad,), -100, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in out.items()}


# ---------- Main ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model + adapter
    ap.add_argument("--base", help="Path to Gemma 4 31B IT base model. Required unless --build-corpus-only.")
    ap.add_argument("--lora-r", type=int, default=256,
                    help="LoRA rank (default 256, 'high rank' per arXiv:2508.05078). Sweep "
                         "candidates: 32, 64, 128, 256, 512.")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha (default = lora-r, i.e., alpha/r = 1.0).")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+", default=None,
                    help="Override LoRA target_modules. Auto-detected for Gemma 4 wrapper "
                         "+ standard Qwen/Llama if omitted.")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Use bitsandbytes 4-bit quantization (QLoRA-style). Required for "
                         "<60GB GPU on 31B model. See docstring memory table.")
    # Corpus
    ap.add_argument("--tasks", nargs="+", default=["arc", "mmlu", "gsm8k", "socialiqa"],
                    help="HF tasks to include in training corpus.")
    ap.add_argument("--n-per-task", type=int, default=500,
                    help="Samples per task when building from HF (default 500 → ~2K total).")
    ap.add_argument("--train-corpus", type=Path, default=None,
                    help="Optional pre-built JSONL corpus path; if omitted, builds from HF.")
    ap.add_argument("--build-corpus-only", action="store_true",
                    help="Build corpus from HF + exit (no training).")
    ap.add_argument("--out-corpus", type=Path, default=None,
                    help="Where to write the built corpus JSONL (default: <out>/train_corpus.jsonl).")
    # Training
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="Effective batch = batch-size * grad-accum (default 1*8=8).")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=1024,
                    help="Max sequence length. Memory scales linearly; lower if OOM at lm_head.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0,
                    help="Save checkpoint every N optimizer steps (0 = end-of-epoch only).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # ---- Corpus build mode (no training) ----
    if args.build_corpus_only:
        _check_deps(need_4bit=False)
        out = args.out_corpus or args.out
        if out.is_dir() or out.suffix != ".jsonl":
            out = (args.out_corpus if args.out_corpus else args.out / "train_corpus.jsonl")
        print(f"=== Build corpus only: tasks={args.tasks} n_per_task={args.n_per_task} ===",
              flush=True)
        build_corpus_from_hf(args.tasks, args.n_per_task, args.seed, out_path=out)
        print(f"=== Corpus built. Exiting (no training requested) ===")
        return

    if args.base is None:
        ap.error("--base is required unless --build-corpus-only is set")

    _check_deps(need_4bit=args.load_in_4bit)

    import torch
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- Load or build corpus ----
    if args.train_corpus is not None:
        print(f"\n=== Phase 1: load pre-built corpus ({args.train_corpus}) ===", flush=True)
        examples = load_corpus_jsonl(args.train_corpus)
    else:
        print(f"\n=== Phase 1: build corpus from HF (tasks={args.tasks}, "
              f"n_per_task={args.n_per_task}) ===", flush=True)
        corpus_out = args.out_corpus or (args.out / "train_corpus.jsonl")
        examples = build_corpus_from_hf(
            args.tasks, args.n_per_task, args.seed, out_path=corpus_out,
        )
    print(f"  loaded {len(examples)} examples")
    task_counts = {}
    for ex in examples:
        task_counts[ex.get("task", "?")] = task_counts.get(ex.get("task", "?"), 0) + 1
    print(f"  per-task: {task_counts}")

    # ---- Tokenizer + base model ----
    print(f"\n=== Phase 2: load base model + tokenizer ({args.base}) ===", flush=True)
    tokenizer = load_tokenizer_compat(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"dtype": torch.bfloat16}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # device_map="auto" required for bitsandbytes quantization placement
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    base_total_params = sum(p.numel() for p in model.parameters())
    print(f"  base model params: {base_total_params/1e9:.2f}B  "
          f"(load_in_4bit={args.load_in_4bit})")

    # Required for gradients to flow through frozen embeddings to LoRA layers
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ---- LoRA config ----
    print(f"\n=== Phase 3: apply LoRA (r={args.lora_r}) ===", flush=True)
    if args.target_modules:
        candidate_targets = args.target_modules
        print(f"  target_modules (manual): {candidate_targets}")
    else:
        candidate_targets = _detect_target_modules(model)
        print(f"  target_modules (auto): {candidate_targets}")
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=candidate_targets,
        # Belt-and-suspenders: even if a target suffix collides with a tower
        # module name, never inject LoRA into the multimodal towers (they are
        # inert on a text corpus -> silent no-op). See A.2 post-mortem 2026-06-02.
        exclude_modules=list(_NON_TEXT_SUBTREES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable LoRA params: {trainable/1e6:.2f}M "
          f"({100*trainable/base_total_params:.3f}% of base)")
    print(f"  gradient checkpointing: enabled")

    # SANITY GUARD (A.2 post-mortem 2026-06-02): fail loudly if the LoRA landed
    # in a multimodal tower instead of the text decoder. On a text corpus a
    # tower-only adapter gets zero gradient -> lora_B stays at zero-init -> the
    # whole run is a silent no-op (flat loss, eval == base). Require that some
    # trainable params live OUTSIDE the towers.
    text_trainable = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and not any(t in n for t in _NON_TEXT_SUBTREES)
    )
    if text_trainable == 0:
        raise RuntimeError(
            "No trainable LoRA params in the text decoder — all landed in "
            f"vision/audio towers ({_NON_TEXT_SUBTREES}). The adapter would be a "
            "no-op on a text corpus. Check target_modules / model structure."
        )
    print(f"  text-decoder trainable params: {text_trainable/1e6:.2f}M "
          f"({100*text_trainable/max(trainable,1):.1f}% of trainable)")

    # ---- Dataset + loader ----
    print(f"\n=== Phase 4: dataset + dataloader ===", flush=True)
    full_ds = AlignLoRADataset(examples, tokenizer, max_length=args.max_length)
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  train: {n_train}, val: {n_val}")

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ---- Optimizer ----
    print(f"\n=== Phase 5: train ({args.epochs} epochs, lr {args.lr}, "
          f"batch {args.batch_size}, accum {args.grad_accum}) ===", flush=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

    history = []
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        t0 = time.time()
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                opt.step()
                opt.zero_grad()
                step += 1
                if args.save_every and step % args.save_every == 0:
                    ckpt_dir = args.out / f"step{step}"
                    model.save_pretrained(ckpt_dir)
                    print(f"    saved checkpoint at step {step}: {ckpt_dir}", flush=True)
            ep_loss += out.loss.item() * batch["input_ids"].size(0)
            ep_n += batch["input_ids"].size(0)
            if (i + 1) % 50 == 0:
                avg_loss = ep_loss / ep_n
                print(f"    ep {ep} step {i+1}/{len(train_loader)}  "
                      f"avg_loss={avg_loss:.4f}  elapsed={time.time()-t0:.1f}s",
                      flush=True)

        # End-of-epoch validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item() * batch["input_ids"].size(0)
                val_n += batch["input_ids"].size(0)
        rec = {
            "epoch": ep,
            "tr_loss": ep_loss / ep_n,
            "val_loss": val_loss / val_n if val_n > 0 else None,
            "elapsed_s": time.time() - t0,
        }
        history.append(rec)
        print(f"  ep {ep}: tr_loss={rec['tr_loss']:.4f}  "
              f"val_loss={rec['val_loss']:.4f}  ({rec['elapsed_s']:.1f}s)", flush=True)

        # Save per-epoch checkpoint + latest pointer (mirror train_router_lora pattern)
        ep_ckpt = args.out / f"ep{ep}"
        model.save_pretrained(ep_ckpt)
        tokenizer.save_pretrained(ep_ckpt)
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)

        log_path = args.out / "train.log.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump({
                "history": history,
                "config": {
                    "base": args.base, "epochs": args.epochs,
                    "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                    "lr": args.lr, "lora_r": args.lora_r, "lora_alpha": lora_alpha,
                    "max_length": args.max_length, "tasks": args.tasks,
                    "n_per_task": args.n_per_task, "n_train": n_train, "n_val": n_val,
                    "load_in_4bit": args.load_in_4bit,
                },
            }, f, indent=2)
        print(f"    saved adapter snapshots: {args.out} (latest) + {ep_ckpt}", flush=True)

    # ---- Smoke test ----
    print(f"\n=== Phase 6: smoke test ===", flush=True)
    model.eval()
    smoke_prompts = [
        ("arc-like", "Question: What gas do plants absorb during photosynthesis?\n"
                     "A. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Helium\nAnswer:"),
        ("gsm8k-like", "Question: Janet has 5 apples. She buys 3 more, then gives 2 "
                       "to her friend. How many does she have?\nAnswer:"),
        ("siqa-like", "Context: Alex helped Sam carry groceries to the car.\n"
                      "Question: How does Sam feel about Alex?\n"
                      "A. Annoyed\nB. Grateful\nC. Confused\nAnswer:"),
    ]
    for label, sp in smoke_prompts:
        messages = [{"role": "user", "content": sp}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(templated, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n[{label}] prompt: {sp[:80]}...")
        print(f"[{label}] output: {gen_text[:200]}")

    print(f"\n=== Done. Adapter at {args.out} ===")
    print(f"Next step: run lm-eval-harness against this adapter for {args.tasks}.")
    print(f"  See scripts/run_lm_eval_v7*.py for the canonical wrapper pattern.")


if __name__ == "__main__":
    main()
