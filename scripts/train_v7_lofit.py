"""V7 LoFiT step 2 — train per-head offset vectors via task-margin SFT.

Pre-requisites:
  1. Head selection probe completed (probe_v7_lofit_head_selection.py).
  2. Top-K heads file produced at runs/probes/v7/p06b_lofit_head_selection.json.

Training:
  For each selected (layer, head) in top-K:
    Trainable parameters:
      α_h ∈ R         (scalar scale, init 1.0)
      θ_h ∈ R^{head_dim}  (offset vector, init 0)
  All base model weights are frozen.
  Forward pre-hook on each selected layer's attn.o_proj injects offsets:
      concat_heads_view[:, :, h, :] += α_h · θ_h
  Loss: task-margin
      L = -log(σ(β · (logp_correct - logp_wrong)))   [DPO-style margin]
    OR L = -logp_correct + γ · logp_wrong            [direct]
  Default: direct margin loss (simpler, no need for β tuning).
  Optimizer: AdamW (CPU- or GPU-side, params are tiny).
  Steps: 500-1000 typical.

Output: runs/v7_lofit/offsets.pt = dict {(layer, head): (alpha, theta)} as fp32 tensors.

Memory budget (RTX 4070 Ti 12GB):
  Base model BF16 frozen: ~10.7 GB
  AdamW state for ~50 heads × (head_dim + 1) ≈ 12k params: ~50 KB
  Per-step activations for forward+backward at seq=384, batch=1: ~500 MB
  Comfortable fit on 4070 Ti.

Usage:
    python -m scripts.train_v7_lofit \\
        --base /path/to/models/gemma4-e2b \\
        --heads-file runs/probes/v7/p06b_lofit_head_selection.json \\
        --top-k 48 \\
        --datasets tqa,arc \\
        --n-train 1500 \\
        --n-val 200 \\
        --steps 800 \\
        --batch-size 1 \\
        --grad-accum 4 \\
        --lr 0.01 \\
        --beta 0.1 \\
        --loss direct \\
        --seq-len-max 384 \\
        --out runs/v7_lofit/offsets.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.lofit_onaxis import frac_perp, map_vinf_to_pairs, onaxis_penalty


# -- Layer / config resolution (shared) --------------------------------------
def _resolve_layers(model) -> list[torch.nn.Module]:
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList):
            return list(layers)
    raise AttributeError("Could not locate decoder layers")


def _resolve_text_config(model):
    cfg = model.config
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


@dataclass
class AttentionShape:
    num_q_heads: int
    head_dim: int


def _get_attention_shape(model, layers=None) -> AttentionShape:
    """Source of truth: o_proj.in_features. Gemma 4 config's num_attention_heads
    can be misleading; derive num_q_heads from the actual o_proj layer."""
    cfg = _resolve_text_config(model)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        num_h = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = hidden // num_h
    if layers is None:
        layers = _resolve_layers(model)
    o_proj = layers[0].self_attn.o_proj
    total_q_dim = o_proj.in_features
    if total_q_dim % head_dim != 0:
        raise AssertionError(
            f"o_proj.in_features={total_q_dim} not divisible by head_dim={head_dim}"
        )
    num_q_heads = total_q_dim // head_dim
    return AttentionShape(num_q_heads=num_q_heads, head_dim=head_dim)


# -- Dataset builders --------------------------------------------------------
# CRITICAL: train and eval scripts MUST use identical splitting logic to avoid
# train-test contamination. Both use seed=0 deterministic permutation, then
# slice at train_frac. split='train' = first train_frac, split='test' = rest.
TRAIN_FRAC = 0.8


def build_tqa_pairs(n: int, split: str = "train", train_frac: float = TRAIN_FRAC, seed: int = 0,
                    wrong_mode: str = "first"):
    """TQA mc1 contrastive pairs. TQA only has 'validation' split natively;
    we deterministically split it into our own train/test using seed-0 perm.

    split='train': first floor(train_frac * n_valid) items
    split='test':  remaining items (disjoint from train by construction)

    n caps the returned size after split selection.

    wrong_mode (audit B-3, 2026-07-23): the HF TQA mc1 dataset lists the gold at
    index 0 and wrong_mode='first' therefore ALWAYS contrasts choices[0] vs
    choices[1] — a fixed positional split, not a sample over distractors. The
    canonical V7-mc offsets were trained with 'first' (kept as default for
    reproducibility); use 'random' for new runs (seeded, samples the wrong
    answer uniformly over all label==0 distractors).
    """
    print(f"  TQA: loading split={split} (target n={n}, wrong_mode={wrong_mode})...", flush=True)
    if wrong_mode == "first":
        print("    WARNING (audit B-3): TQA gold is dataset-position-0, so wrong_mode='first' "
              "always pairs choices[0] vs choices[1]. Prefer wrong_mode='random' for new runs.",
              flush=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    wrong_rng = torch.Generator().manual_seed(seed + 7919)

    all_valid = []
    for i in perm:
        ex = ds[i]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        if 1 not in labels:
            continue
        cidx = labels.index(1)
        wrong = [j for j, l in enumerate(labels) if l == 0]
        if not wrong:
            continue
        if wrong_mode == "random":
            widx = wrong[int(torch.randint(len(wrong), (1,), generator=wrong_rng))]
        else:
            widx = wrong[0]
        all_valid.append((ex["question"], choices[cidx], choices[widx]))

    n_total = len(all_valid)
    n_train = int(n_total * train_frac)
    if split == "train":
        chosen = all_valid[:n_train]
    elif split == "test":
        chosen = all_valid[n_train:]
    else:
        raise ValueError(f"Unknown split: {split}")
    chosen = chosen[:n]
    print(f"    TQA total valid={n_total}, train={n_train}, test={n_total-n_train}, returning {len(chosen)}")
    return chosen


def build_arc_pairs(n: int, split: str = "train", seed: int = 0):
    """ARC contrastive pairs. ARC-Challenge has native train/test split; we use
    those directly (already disjoint by construction).
    """
    hf_split = "train" if split == "train" else "test"
    print(f"  ARC: loading {hf_split} split (target n={n})...", flush=True)
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=hf_split)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    pairs = []
    for i in perm:
        ex = ds[i]
        ans = ex["answerKey"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        if ans not in labels:
            continue
        cidx = labels.index(ans)
        wrong = [j for j, l in enumerate(labels) if l != ans]
        if not wrong:
            continue
        pairs.append((ex["question"], choices[cidx], choices[wrong[0]]))
        if len(pairs) >= n:
            break
    print(f"    ARC {hf_split}: returning {len(pairs)} pairs")
    return pairs


def build_lambada_pairs(n: int, split: str = "train", train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Lambada contrastive pairs for completion-LoFiT training.

    Lambada has only 'test' split (5153 samples). Deterministic train/test
    split using seed=0 perm + train_frac, mirroring TQA logic.

    pair = (prompt, gold_last_word, wrong_word)
    wrong_word is sampled from another sample's last word (plausible English,
    similar register; ensures contrastive signal isn't just "real word vs garbage").
    """
    print(f"  Lambada: loading split={split} (target n={n})...", flush=True)
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()

    all_valid = []
    for i in perm:
        text = ds[i]["text"].strip()
        words = text.split()
        if len(words) < 4:
            continue
        prompt = " ".join(words[:-1])
        gold = words[-1]
        all_valid.append((prompt, gold))

    n_total = len(all_valid)
    n_train = int(n_total * train_frac)
    if split == "train":
        candidates = all_valid[:n_train]
    elif split == "test":
        candidates = all_valid[n_train:]
    else:
        raise ValueError(f"Unknown split: {split}")
    candidates = candidates[:n]

    pool = [g for _, g in candidates]
    pairs = []
    for i, (prompt, gold) in enumerate(candidates):
        wrong = pool[(i + 7) % len(pool)]
        if wrong.lower() == gold.lower():
            wrong = pool[(i + 13) % len(pool)]
        # Continuation = " <word>" so tokenizer sees a leading space (mirrors mc convention)
        pairs.append((prompt, " " + gold, " " + wrong))
    print(f"    Lambada {split}: returning {len(pairs)} pairs (total={n_total}, train={n_train})")
    return pairs


def build_gsm8k_pairs(n: int, split: str = "train", seed: int = 0):
    """GSM8K contrastive pairs for reasoning-LoFiT training.

    GSM8K has native train/test splits — use directly.

    pair = (prompt, " <gold_number>", " <wrong_number>")
    wrong is gold ± small offset OR another problem's gold. The ±1/±2/±3
    perturbation forces the per-head signal to discriminate close numbers,
    which is the actual reasoning challenge.
    """
    import re
    hf_split = "train" if split == "train" else "test"
    print(f"  GSM8K: loading {hf_split} split (target n={n})...", flush=True)
    ds = load_dataset("gsm8k", "main", split=hf_split)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    candidates = []
    for i in perm:
        ex = ds[i]
        question = ex["question"].strip()
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ex["answer"])
        if m is None:
            continue
        gold = m.group(1)
        candidates.append((question, gold))
        if len(candidates) >= n:
            break

    pool = [g for _, g in candidates]
    pairs = []
    for i, (q, gold) in enumerate(candidates):
        if i % 2 == 0:
            try:
                gv = float(gold)
                offset = ((i % 3) + 1) * (1 if (i % 4) < 2 else -1)
                wrong_v = gv + offset
                wrong = str(int(wrong_v)) if "." not in gold else f"{wrong_v}"
            except ValueError:
                wrong = pool[(i + 7) % len(pool)]
        else:
            wrong = pool[(i + 11) % len(pool)]
            if wrong == gold:
                wrong = pool[(i + 17) % len(pool)]
        prompt = "Question: " + q + "\nAnswer:"
        pairs.append((prompt, " " + gold, " " + wrong))
    print(f"    GSM8K {hf_split}: returning {len(pairs)} pairs")
    return pairs


def build_hellaswag_pairs(n: int, split: str = "train", train_frac: float = TRAIN_FRAC, seed: int = 0):
    """HellaSwag contrastive pairs for narrative-continuation LoFiT training.

    HellaSwag has a 'validation' split with labels (10k items). 'train' has no
    labels. We use validation, deterministic train/test split via seed=0 perm.

    pair = (ctx, " gold_ending", " wrong_ending")
    Wrong = FIRST non-gold ending (adversarial-generated, plausible).
    """
    print(f"  HellaSwag: loading split={split} (target n={n})...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()

    all_valid = []
    for i in perm:
        ex = ds[i]
        label_str = ex.get("label", "")
        if label_str == "":
            continue
        try:
            gold = int(label_str)
        except (TypeError, ValueError):
            continue
        ctx = (ex["ctx_a"] or "").strip()
        if ex.get("ctx_b"):
            ctx_b = ex["ctx_b"].strip()
            if ctx_b:
                ctx = f"{ctx} {ctx_b[0].upper()}{ctx_b[1:]}" if ctx else ctx_b
        endings = ex["endings"]
        if gold >= len(endings):
            continue
        wrongs = [j for j in range(len(endings)) if j != gold]
        if not wrongs:
            continue
        all_valid.append((ctx, " " + endings[gold].strip(), " " + endings[wrongs[0]].strip()))

    n_total = len(all_valid)
    n_train = int(n_total * train_frac)
    if split == "train":
        chosen = all_valid[:n_train]
    elif split == "test":
        chosen = all_valid[n_train:]
    else:
        raise ValueError(f"Unknown split: {split}")
    chosen = chosen[:n]
    print(f"    HellaSwag {split}: returning {len(chosen)} pairs (total={n_total}, train={n_train})")
    return chosen


def _load_phase3_jsonl(jsonl_path: str, n: int, split: str,
                       train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Load Phase 3 contrastive pairs from a JSONL file (one obj per line with
    keys: prompt, correct, wrong[, source]). Deterministic train/test split via
    seed=0 perm + train_frac, mirroring TQA logic. Adds " " (space) prefix to
    correct/wrong to match the leading-space convention used by the other
    builders (TQA returns raw strings; lambada/gsm8k/hellaswag prepend " ").
    For chat-template training we pass separator="" so the leading space here
    is what places the answer one token after the turn marker.
    """
    print(f"  Phase3 JSONL [{jsonl_path}]: loading split={split} (target n={n})...", flush=True)
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Phase 3 pairs file not found: {jsonl_path}. Run the corresponding "
            f"data-generation script (e.g. scripts/build_coding_pairs.py) first."
        )
    all_rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        all_rows.append((obj["prompt"], obj["correct"], obj["wrong"]))

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_rows), generator=rng).tolist()
    shuffled = [all_rows[i] for i in perm]

    n_total = len(shuffled)
    n_train = int(n_total * train_frac)
    if split == "train":
        chosen = shuffled[:n_train]
    elif split == "test":
        chosen = shuffled[n_train:]
    else:
        raise ValueError(f"Unknown split: {split}")
    chosen = chosen[:n]
    pairs = [(q, " " + c, " " + w) for q, c, w in chosen]
    print(f"    Phase3 [{jsonl_path}] {split}: returning {len(pairs)} pairs "
          f"(total={n_total}, train={n_train})")
    return pairs


def build_phase3_coding_pairs(n: int, split: str = "train",
                              train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Phase 3 Path A coding adapter contrastive pairs. Loads from
    runs/phase3_adapters/coding/train/pairs.jsonl (HumanEval/MBPP correct +
    Sonnet-generated buggy variants). See scripts/build_coding_pairs.py."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/coding/train/pairs.jsonl",
        n, split, train_frac, seed,
    )


def build_phase3_proof_pairs(n: int, split: str = "train",
                             train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Phase 3 Path C math_proof_or_format adapter contrastive pairs. Loads from
    runs/phase3_adapters/math_proof/train/pairs.jsonl (proof-step corruption +
    formatted-output pairs). See scripts/build_proof_pairs.py."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/math_proof/train/pairs.jsonl",
        n, split, train_frac, seed,
    )


def build_phase6_g_socialiqa_pairs(n: int, split: str = "train",
                                    train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Phase 6 G adapter (SocialIQA ToM) contrastive pairs. Loads from
    runs/phase3_adapters/socialiqa/train/pairs.jsonl (lighteval/siqa correct
    answer vs distractor choices, 2 pairs per question). See
    scripts/build_socialiqa_pairs.py."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/socialiqa/train/pairs.jsonl",
        n, split, train_frac, seed,
    )


def build_phase1c_kr_sciq_pairs(n: int, split: str = "train",
                                  train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Phase 1.C Item 1.5 KR-dedicated adapter (SciQ science MC) contrastive
    pairs. Loads from runs/phase3_adapters/kr_sciq/train/pairs.jsonl (allenai/sciq
    correct_answer vs distractor1/2/3, 3 pairs per question). See
    scripts/build_sciq_pairs.py + feedback_item_1_5_corpus_survey_2026_05_25."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/kr_sciq/train/pairs.jsonl",
        n, split, train_frac, seed,
    )


def build_phase1c_kr_opcion_d_pairs(n: int, split: str = "train",
                                      train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Phase 1.C Item 1.5 Opción D KR-mix multi-corpus contrastive pairs.
    Loads from runs/phase3_adapters/kr_opcion_d/train/pairs.jsonl (shuffled
    mix of SciQ ~2100 + MedMCQA ~2800 + head_qa_v2 ~1400 ≈ 6300-7000 total).
    Pivot from single-corpus SciQ after K=24 SciQ transfer-gap verdict
    (sciq +2.1pp / GPQA Diamond -12.1pp). See scripts/build_kr_opcion_d_mix.py
    + feedback_item_1_5_k24_sciq_overshoot_verdict_2026_05_26."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/kr_opcion_d/train/pairs.jsonl",
        n, split, train_frac, seed,
    )


def build_phase1c_kr_medmcqa_pairs(n: int, split: str = "train",
                                     train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Per-corpus train reader for diagnostic / future per-corpus adapters
    (Opción D triage 2026-05-27)."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/kr_opcion_d/train/medmcqa_pairs.jsonl",
        n, split, train_frac, seed,
    )


def build_phase1c_kr_head_qa_v2_pairs(n: int, split: str = "train",
                                        train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Per-corpus train reader (Opción D triage 2026-05-27)."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/kr_opcion_d/train/head_qa_pairs.jsonl",
        n, split, train_frac, seed,
    )


def build_phase1c_kr_ewok_pairs(n: int, split: str = "train",
                                  train_frac: float = TRAIN_FRAC, seed: int = 0):
    """Per-corpus train reader (Opción D triage 2026-05-27). NOTE EWoK uses raw
    Context/Target prompts, NOT Question:/Answer: template — formatting confound
    candidate H4 for mix-probe failure."""
    return _load_phase3_jsonl(
        "runs/phase3_adapters/kr_opcion_d/train/ewok_pairs.jsonl",
        n, split, train_frac, seed,
    )


DATASETS = {
    "tqa": build_tqa_pairs,
    "arc": build_arc_pairs,
    "lambada": build_lambada_pairs,
    "gsm8k": build_gsm8k_pairs,
    "hellaswag": build_hellaswag_pairs,
    # Phase 3 builders — load Sonnet-augmented JSONL written by the
    # scripts/build_{coding,proof}_pairs.py data-generation scripts.
    "phase3_coding": build_phase3_coding_pairs,
    "phase3_proof": build_phase3_proof_pairs,
    # Phase 6 G adapter — SocialIQA ToM (first expansion of the Mu adapter
    # roster beyond {A.2 coding, B.2 mc_disc, C.3 math_proof_mid_only}).
    "phase6_g_socialiqa": build_phase6_g_socialiqa_pairs,
    # Phase 1.C Item 1.5 — KR-dedicated adapter (SciQ Phys/Chem/Bio MC). KR
    # is the 4th cognitive-mode primary (D/C/I/KR per v2.1 framework).
    "phase1c_kr_sciq": build_phase1c_kr_sciq_pairs,
    # Phase 1.C Item 1.5 Opción D — multi-corpus KR mix (SciQ + MedMCQA +
    # head_qa_v2). Pivot post K=24 SciQ overshoot verdict 2026-05-26.
    "phase1c_kr_opcion_d": build_phase1c_kr_opcion_d_pairs,
    # Per-corpus diagnostic / future per-corpus-adapter readers
    # (Opción D triage 2026-05-27).
    "phase1c_kr_medmcqa": build_phase1c_kr_medmcqa_pairs,
    "phase1c_kr_head_qa_v2": build_phase1c_kr_head_qa_v2_pairs,
    "phase1c_kr_ewok": build_phase1c_kr_ewok_pairs,
}


# -- LoFiT offset module ------------------------------------------------------
class LofitOffsets(torch.nn.Module):
    """Trainable per-(layer, head) offset bank.

    Stores α_h and θ_h for all selected heads. Provides accessors for the
    forward pre-hook to add offsets at the right places.
    """

    def __init__(
        self, layer_head_pairs: list[tuple[int, int]], head_dim: int,
        device: str, dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.layer_head_pairs = list(layer_head_pairs)
        self.head_dim = head_dim
        self.device_ = device
        self.dtype_ = dtype

        # Group pairs by layer for hook efficiency.
        self.by_layer: dict[int, list[int]] = {}
        for li, hi in self.layer_head_pairs:
            self.by_layer.setdefault(li, []).append(hi)

        # Parameters: each pair gets its own α and θ.
        # Index: (li, hi) → flat index in self.params.
        n_pairs = len(self.layer_head_pairs)
        self.alpha = torch.nn.Parameter(torch.ones(n_pairs, dtype=dtype, device=device))
        self.theta = torch.nn.Parameter(torch.zeros(n_pairs, head_dim, dtype=dtype, device=device))

        # Reverse lookup
        self.pair_to_idx: dict[tuple[int, int], int] = {
            (li, hi): idx for idx, (li, hi) in enumerate(self.layer_head_pairs)
        }

        # Pre-computed per-layer LongTensor index buffers for vectorized hooks.
        # `_layer_pair_indices[li]` = flat indices into alpha/theta of layer li's pairs.
        # `_layer_head_indices[li]` = head positions (0..num_q_heads-1) at layer li.
        # Stored as tensors (not Python lists) so the hook hot-path performs a single
        # advanced-index gather instead of a per-head Python loop. This is the
        # vectorization path consumed by `install_lofit_hooks` below.
        self._layer_pair_indices: dict[int, torch.Tensor] = {}
        self._layer_head_indices: dict[int, torch.Tensor] = {}
        for li, hi_list in self.by_layer.items():
            pair_idxs = [self.pair_to_idx[(li, hi)] for hi in hi_list]
            self._layer_pair_indices[li] = torch.tensor(
                pair_idxs, dtype=torch.long, device=device
            )
            self._layer_head_indices[li] = torch.tensor(
                hi_list, dtype=torch.long, device=device
            )

    def get_layer_offsets(self, layer_idx: int) -> dict[int, torch.Tensor]:
        """Return {head_idx: alpha * theta} for all trainable heads at this layer.

        Legacy accessor (pre-vectorization). Retained for any external script that
        introspects the offset values per-head. The vectorized hook (used by
        `install_lofit_hooks`) goes through `get_layer_indices` instead.
        """
        if layer_idx not in self.by_layer:
            return {}
        out = {}
        for hi in self.by_layer[layer_idx]:
            idx = self.pair_to_idx[(layer_idx, hi)]
            out[hi] = self.alpha[idx] * self.theta[idx]  # (head_dim,)
        return out

    def get_layer_indices(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pair_indices, head_indices) LongTensors for the given layer.

        Used by the vectorized hook to gather α/θ slices and scatter offsets into
        a per-layer workspace in a single advanced-index op. Returns (empty, empty)
        if the layer has no selected heads.
        """
        if layer_idx not in self._layer_pair_indices:
            empty = torch.empty(0, dtype=torch.long, device=self.device_)
            return empty, empty
        return self._layer_pair_indices[layer_idx], self._layer_head_indices[layer_idx]


# -- Hook installation -------------------------------------------------------
def install_lofit_hooks(layers, offsets: LofitOffsets, attn_shape: AttentionShape):
    """Register pre-hooks on attn.o_proj for each layer with selected heads.

    Vectorized: gathers α/θ slices for the layer in a single advanced-index op,
    computes α·θ via broadcast multiply, scatters into a small
    (num_q_heads, head_dim) workspace, and broadcast-adds to concat. The hot
    path runs ~4 CUDA kernels per layer per token regardless of K, replacing
    the prior Python loop that issued ~K kernels per layer per token.

    Numerical identity vs the prior implementation: the per-coordinate value
    of `modified[b, t, hi*head_dim + d]` is `concat[...] + α_h · θ_h` for
    selected (li, hi) and `concat[...]` otherwise — identical in both
    implementations under any input shape / dtype.

    Grad flow: `layer_offset[h_idx] = active` records a CopySlices node;
    backward correctly accumulates into alpha and theta.

    Defensive shape check: Gemma 4 has alternating sliding/global attention
    where global layers have 2× the q-dim. If a selected head comes from a
    non-matching layer, the reshape would fail. We raise a clear error.
    """
    expected_total = attn_shape.num_q_heads * attn_shape.head_dim
    num_q_heads = attn_shape.num_q_heads
    head_dim = attn_shape.head_dim
    handles = []
    for layer_idx in offsets.by_layer:
        layer = layers[layer_idx]
        o_proj = layer.self_attn.o_proj
        pair_idx, head_idx = offsets.get_layer_indices(layer_idx)

        def make_hook(li, p_idx, h_idx):
            def hook(_module, inputs):
                concat = inputs[0]  # (B, T, total)
                B, T, total = concat.shape
                if total != expected_total:
                    raise RuntimeError(
                        f"Layer {li}: o_proj input dim {total} != expected "
                        f"{expected_total} ({num_q_heads} × {head_dim}). "
                        f"This layer has different attention structure (likely "
                        f"global vs sliding). Selected head file should not "
                        f"include heads from this layer."
                    )
                # Vectorized: gather α and θ slices for this layer, multiply,
                # cast to model dtype, scatter into a tiny (num_q_heads, head_dim)
                # workspace, then broadcast-add. No Python loop in the hot path.
                # Grad flows via setitem CopySlices into alpha/theta.
                active = (offsets.alpha[p_idx].unsqueeze(-1) * offsets.theta[p_idx]).to(
                    dtype=concat.dtype
                )  # (n_layer_heads, head_dim)
                layer_offset = concat.new_zeros(num_q_heads, head_dim)
                layer_offset[h_idx] = active
                modified = concat + layer_offset.view(total)  # broadcast (B, T, total) + (total,)
                return (modified,) + inputs[1:]
            return hook

        handles.append(
            o_proj.register_forward_pre_hook(make_hook(layer_idx, pair_idx, head_idx))
        )
    return handles


def detect_layer_dims(model, tokenizer, layers, expected_total: int) -> dict:
    """Run one forward pass and record o_proj input dims per layer.

    Used to validate that selected heads come from layers compatible with the
    expected attention shape.
    """
    detected: dict[int, int] = {}
    handles = []
    for li, layer in enumerate(layers):
        o_proj = layer.self_attn.o_proj

        def make_hook(layer_idx):
            def hook(_module, inputs):
                detected[layer_idx] = inputs[0].shape[-1]
            return hook

        handles.append(o_proj.register_forward_pre_hook(make_hook(li)))

    toks = tokenizer("Hello world", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        model(**toks, use_cache=False)

    for h in handles:
        h.remove()
    return detected


def remove_handles(handles):
    for h in handles:
        h.remove()


# -- P9 on-axis penalty wiring (training-time on-axis constraint) -------------
# The constructive flip: force the trainable offset alpha*theta to live ON the functional
# inference axis v_inf by penalizing its off-axis mass. Math in scripts/lofit_onaxis.py.
def load_vinf_for_pairs(vinf_path, layer_head_pairs, device="cpu"):
    """Load functional_directions.pt and map v_inf onto the offset's (layer, head) pairs.

    Returns (v_inf_perpair (n_pairs, head_dim), mask (n_pairs,)) on `device`. Masked pairs
    (no captured v_inf for that model layer) get a zero row + mask 0.
    """
    fd = torch.load(vinf_path, map_location="cpu", weights_only=False)
    v_per, mask = map_vinf_to_pairs(fd, layer_head_pairs)
    return v_per.to(device), mask.to(device)


def onaxis_penalty_term(offsets: "LofitOffsets", v_inf_perpair, mask, lam: float):
    """Scaled on-axis penalty for the current offset params, ready to `.backward()`.

    lam * (mean off-axis energy of alpha*theta projected against v_inf). Differentiable in
    offsets.alpha and offsets.theta. Added once per optimizer step (a regularizer on the
    params, not a per-sample term): L_total = L_margin + lam * L_onaxis.
    """
    return lam * onaxis_penalty(offsets.alpha, offsets.theta, v_inf_perpair, mask)


# -- Loss computation --------------------------------------------------------
def continuation_logprob(
    model, tokenizer, prompt: str, continuation: str,
    seq_len_max: int, device: str, separator: str = " ",
) -> torch.Tensor:
    """Differentiable: returns sum log P(continuation | prompt).

    separator: inserted between prompt and continuation. Default " " (space) for
    raw text Q:/A: format. For chat template (prompt ends with turn marker /
    newline already), pass separator="" to avoid double-spacing.
    """
    full = prompt + separator + continuation
    full_ids = tokenizer(full, return_tensors="pt", truncation=True,
                         max_length=seq_len_max).input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=seq_len_max).input_ids.to(device)
    n_p = prompt_ids.shape[1]
    n_f = full_ids.shape[1]
    if n_f <= n_p:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits = model(input_ids=full_ids, use_cache=False).logits[0]  # (seq, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    cont_token_ids = full_ids[0, n_p:n_f]
    rel = log_probs[n_p - 1 : n_f - 1]
    tok_lp = rel.gather(1, cont_token_ids.unsqueeze(1)).squeeze(1)
    return tok_lp.sum()


def _build_prompt_chat(tokenizer, q: str) -> str:
    """Wrap question in chat-template prefix with add_generation_prompt=True.
    Last token of the resulting text is the assistant turn marker; concat answer
    directly to land at last-answer-token capture position.

    See feedback_v7_chat_template_train_deploy_gap.md for rationale.
    """
    messages = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def task_margin_loss(
    model, tokenizer, pair: tuple[str, str, str],
    seq_len_max: int, device: str, loss_type: str, beta: float = 0.1,
    gamma: float = 0.5, chat_template: bool = False,
) -> torch.Tensor:
    """One pair → scalar loss.

    loss_type='direct':   L = -logp_correct + gamma * logp_wrong
    loss_type='plain_ce': L = -logp_correct  (non-contrastive control, E1)
    loss_type='dpo':       L = -log(sigmoid(beta * (logp_correct - logp_wrong)))

    chat_template=True: wrap q in chat-template prefix (for IT models like
    Gemma 4 IT, Qwen IT). Required for deploy-relevant V7 training — V7 trained
    on raw text Q:/A: format does not transfer to chat-template inference.
    """
    q, correct, wrong = pair
    if chat_template:
        prompt = _build_prompt_chat(tokenizer, q)
        # Chat template ends with turn marker / newline — no separator needed
        lp_c = continuation_logprob(model, tokenizer, prompt, correct, seq_len_max, device, separator="")
        lp_w = continuation_logprob(model, tokenizer, prompt, wrong, seq_len_max, device, separator="")
    else:
        prompt = f"Question: {q}\nAnswer:"
        lp_c = continuation_logprob(model, tokenizer, prompt, correct, seq_len_max, device)
        lp_w = continuation_logprob(model, tokenizer, prompt, wrong, seq_len_max, device)

    if loss_type == "direct":
        return -lp_c + gamma * lp_w
    elif loss_type == "anti_direct":
        # Anti-objective (E7, 2026-07-25): 'direct' with correct<->wrong swapped, i.e.
        # train the SAME 48 heads to prefer the distractor. Not a control for the
        # parameterization (that is plain_ce/E1) — it asks a different question: does
        # the optimizer converge to -theta_mc, or does it find another shortcut?
        # NOTE: val_acc DROPS under this objective, so the trainer's best-checkpoint
        # selection is inverted here. Use the .final.pt blob, never the 'best' one.
        return -lp_w + gamma * lp_c
    elif loss_type == "plain_ce":
        # Non-contrastive control (E1, external-review remediation 2026-07-24):
        # ordinary CE / NLL on the correct answer only, NO term against the wrong
        # option. Same per-head offset parameterization as 'direct' (identical
        # 12,336 params, same 48 heads) — isolates the contrastive objective from
        # the parameterization. lp_w is computed above but unused here: this keeps
        # the diff minimal and regression-safe (direct/dpo untouched); the cost is
        # one extra, discarded forward per pair.
        return -lp_c
    elif loss_type == "dpo":
        margin = beta * (lp_c - lp_w)
        return -F.logsigmoid(margin)
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")


# -- Trajectory logging for mid-training downstream eval -------------------
@dataclass
class TrajectoryEntry:
    """One row of mid-training trajectory log.

    Schema definition for `<out>.trajectory.json` (Polish item #3). When
    a cycle fails (lm_eval raised), `downstream_primary_acc` and
    `downstream_spillover_accs` are None — distinguishes "this cycle
    failed" from "key missing" (which would silently corrupt analyses).

    Written atomically (write-temp-then-rename) after EACH cycle so a
    mid-training crash never loses accumulated log (Polish item #4).
    """
    step: int
    val_loss: float
    val_acc: float
    downstream_primary_acc: float | None        # None if cycle eval failed
    downstream_spillover_accs: dict[str, float | None]  # {task: acc or None}
    ckpt_path: str                              # absolute path to .step_N.pt
    elapsed_s: float
    is_best_primary: bool                       # True if this cycle's primary beats prior best

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "downstream_primary_acc": self.downstream_primary_acc,
            "downstream_spillover_accs": self.downstream_spillover_accs,
            "ckpt_path": self.ckpt_path,
            "elapsed_s": self.elapsed_s,
            "is_best_primary": self.is_best_primary,
        }


def _save_trajectory_atomic(path: Path, entries: list[TrajectoryEntry], meta: dict) -> None:
    """Atomic write of trajectory log + meta. Call after EVERY cycle so
    a crash mid-training preserves all prior cycles (Polish item #4).
    """
    payload = {
        "schema_version": "1.0",
        "meta": meta,
        "entries": [e.to_dict() for e in entries],
    }
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _validate_downstream_tasks(primary: str, spillover: list[str]) -> None:
    """Fail-fast task validation at startup so a typo doesn't burn an
    hour of training. Lazy-import lm_eval (Polish item #5) so the
    backwards-compat path (--downstream-primary empty) doesn't require
    lm_eval to be installed.
    """
    if not primary:
        return  # feature off, nothing to validate

    try:
        from lm_eval.tasks import TaskManager
    except ImportError as e:
        raise SystemExit(
            f"--downstream-primary set ({primary!r}) but lm_eval is not "
            f"installed. Run `pip install 'lm-eval[hf]'` or unset the flag. "
            f"Original error: {e}"
        )

    tm = TaskManager()
    available = set(tm.all_tasks)
    tasks_to_check = [primary] + spillover
    missing = [t for t in tasks_to_check if t not in available]
    if missing:
        raise SystemExit(
            f"Unknown lm_eval task(s): {missing}. "
            f"Run `lm_eval --tasks list` to see available tasks."
        )


def _run_downstream_eval(
    model,
    tokenizer,
    primary: str,
    spillover: list[str],
    limit: int,
    apply_chat_template: bool,
) -> dict[str, float | None]:
    """Run lm_eval.simple_evaluate over [primary, *spillover] tasks on
    a subset of `limit` items each. Returns dict {task: acc_norm or None
    on failure}. The model is wrapped in HFLM(pretrained=<instance>)
    so forward pre-hooks installed by the trainer are preserved (verified
    in test_hflm_preserves_forward_pre_hooks, Polish item #1).

    Defensive limit clamp (Polish item #6): lm_eval's limit param accepts
    larger-than-task-size silently; logging clamped value avoids confusion.

    Restores model.train() after eval (Polish item #7) — simple_evaluate
    may call model.eval() internally and we want training to resume
    in train mode (matters if any future submodule has dropout).
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = [primary] + spillover
    was_training = model.training

    try:
        # Wrap the already-loaded model + hooks in HFLM
        hflm = HFLM(pretrained=model, tokenizer=tokenizer)
        results = simple_evaluate(
            model=hflm,
            tasks=tasks,
            limit=limit,
            apply_chat_template=apply_chat_template,
            log_samples=False,           # avoid bloating logs mid-training
            verbosity="ERROR",           # suppress lm_eval INFO chatter
            bootstrap_iters=0,           # skip stderr bootstrap (saves ~5s)
        )
    except Exception as e:
        print(f"[downstream-eval] FAILED: {type(e).__name__}: {e}")
        # Restore train mode and return None for all tasks
        if was_training:
            model.train()
        return {t: None for t in tasks}

    # Restore train mode (Polish item #7)
    if was_training:
        model.train()

    # Extract acc_norm per task
    out = {}
    task_results = results.get("results", {})
    for t in tasks:
        task_data = task_results.get(t, {})
        # Prefer acc_norm, fall back to acc
        acc = task_data.get("acc_norm,none",
              task_data.get("acc_norm",
              task_data.get("acc,none",
              task_data.get("acc"))))
        out[t] = float(acc) if acc is not None else None

    return out


# -- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--heads-file", required=True,
                    help="JSON output of probe_v7_lofit_head_selection.py")
    ap.add_argument("--top-k", type=int, default=48,
                    help="Use top-K heads from heads-file (overrides if smaller than file's ranked_top_k)")
    ap.add_argument("--datasets", default="tqa,arc")
    ap.add_argument("--n-train", type=int, default=1500,
                    help="Pairs per dataset for training")
    ap.add_argument("--n-val", type=int, default=200,
                    help="Held-out pairs per dataset for validation")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="Weight on logp_wrong term in 'direct' loss")
    ap.add_argument("--loss", default="direct",
                    choices=["direct", "dpo", "plain_ce", "anti_direct"],
                    help="'anti_direct' (E7) swaps correct<->wrong in 'direct'. Under it "
                         "val_acc falls by design and best-checkpoint selection inverts: "
                         "take the .final.pt blob, not the 'best' one.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Batch-visit-order RNG seed (line ~1236). Init is deterministic "
                         "(alpha=1, theta=0), so with a fixed --data-seed this varies ONLY "
                         "batch-visit order. E1 (non-contrastive control) holds --data-seed "
                         "fixed and sweeps this for same-data optimization-variance seeds.")
    ap.add_argument("--data-seed", type=int, default=0,
                    help="Train/val split RNG seed (per-dataset train-pool permutation). "
                         "Default 0 = canonical split (byte-identical to the released run). "
                         "E1 keeps this 0 ('same examples', per the review); E2 varies it "
                         "with --seed for full data-resampling replicates.")
    ap.add_argument("--seq-len-max", type=int, default=384)
    ap.add_argument("--eval-every", type=int, default=200,
                    help="Run val every N steps. Bumped to 200 default 2026-05-11 "
                         "to reduce eval overhead on Gemma 4 31B BF16 / 96GB Blackwell; "
                         "val_pairs (cap 25 per cycle) cost ~10-20s on 31B and we don't "
                         "need finer monitoring resolution. Set 100 for fine-grained probe.")
    ap.add_argument("--attn-impl", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="Attention implementation. Default 'sdpa' is safe on all hardware "
                         "and head_dims. 'flash_attention_2' may be ~10-25%% faster on long "
                         "seq lengths BUT fails at runtime on Gemma 4 (head_dim=256) on some "
                         "FA2 builds even when the load succeeds — observed 2026-05-11 on "
                         "Blackwell. Use 'eager' only for debugging hook interactions.")
    ap.add_argument("--chat-template", action="store_true",
                    help="Use chat template wrap for IT models. REQUIRED for chat-aligned "
                         "deploy: V7 trained on raw text Q:/A: does not transfer to chat-template "
                         "inference. Builds prompts via apply_chat_template + add_generation_prompt. "
                         "See feedback_v7_chat_template_train_deploy_gap.md.")
    ap.add_argument("--out", required=True)
    # ---- P9: training-time on-axis constraint (penalty toward v_inf) ----
    # Off by default (behavior identical to pre-P9). When on, adds a projection penalty
    # that forces the trainable offset alpha*theta ON-axis (toward v_inf) by penalizing its
    # off-axis mass: L = L_margin + onaxis_lambda * L_onaxis. Sweep lambda on a log grid;
    # lambda=0 = off-axis baseline. The gate this feeds is scripts/analyze_p9_gate.py.
    ap.add_argument("--onaxis-penalty", action="store_true",
                    help="P9: enable the on-axis projection penalty toward v_inf.")
    ap.add_argument("--onaxis-lambda", type=float, default=0.0,
                    help="P9: weight on the on-axis penalty (default 0.0 = no penalty).")
    ap.add_argument("--vinf-path", default="runs/functional_directions.pt",
                    help="P9: functional_directions.pt providing v_inf + captured_indices.")
    # ---- Mid-training downstream eval (opt-in feature) ----
    # When --downstream-primary is empty (default), the script behaves
    # identically to pre-feature: only val_pair_acc tracked, only
    # `<out>.pt` (best) and `<out>.final.pt` written. When set, each
    # eval cycle also runs lm_eval.simple_evaluate over a subset of the
    # specified task(s) and selects best ckpt by downstream acc_norm.
    ap.add_argument("--downstream-primary", default="",
                    help="lm_eval task name (e.g., 'sciq') driving best-ckpt "
                         "selection. Empty (default) = feature off; behavior "
                         "identical to pre-2026-05-26. See spec "
                         "2026-05-26-train-v7-lofit-midtrain-eval-design.md.")
    ap.add_argument("--downstream-spillover", default="",
                    help="Comma-separated lm_eval task names monitored each "
                         "eval cycle, but NOT used for best-ckpt selection. "
                         "WARN logged if any drops > --downstream-spillover-warn-pp "
                         "vs first cycle. Example: 'mmlu,arc_challenge'.")
    ap.add_argument("--downstream-limit", type=int, default=50,
                    help="Per-task subset size for mid-training eval (lm_eval "
                         "limit=N: deterministic first-N items). Default 50.")
    ap.add_argument("--downstream-eval-every", type=int, default=None,
                    help="Run downstream eval every N steps. Default = same "
                         "as --eval-every. Set higher to reduce overhead.")
    ap.add_argument("--downstream-apply-chat-template",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Apply chat template during downstream eval. "
                         "REQUIRED for IT base models. Default True. "
                         "Use --no-downstream-apply-chat-template to disable.")
    ap.add_argument("--downstream-spillover-warn-pp", type=float, default=3.0,
                    help="Spillover acc drop (in pp vs first cycle) above "
                         "which a WARN is logged. Default 3.0 (per D.1 lesson).")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log.json")

    print("=" * 60)
    print("V7 LoFiT step 2 — train per-head offsets")
    print("=" * 60)
    print(f"  base: {args.base}")
    print(f"  heads_file: {args.heads_file}")
    print(f"  top-k: {args.top_k}")
    print(f"  datasets: {args.datasets}")
    print(f"  n_train: {args.n_train}, n_val: {args.n_val}")
    print(f"  steps: {args.steps}, batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"  lr: {args.lr}, loss: {args.loss}")
    if args.loss == "anti_direct":
        print()
        print("  !!! ANTI-OBJECTIVE (E7): training the heads to prefer the WRONG option.")
        print("      val_acc is EXPECTED to fall; the 'best' checkpoint selection is")
        print(f"      inverted under this loss. Use {out_path.with_suffix('.final.pt').name},")
        print("      NOT the best-val blob.")
    print()

    # Load heads
    heads_data = json.loads(Path(args.heads_file).read_text())
    ranked = heads_data["ranked_top_k"]
    selected_raw = ranked[: args.top_k]
    layer_head_pairs_raw = [(h["layer"], h["head"]) for h in selected_raw]
    print(f"  Loaded top-{args.top_k} heads from file (will filter to compatible layers)")
    print()

    # Build datasets — clean splits, train/test disjoint by construction.
    #
    # val construction (fixed 2026-07-23, audit B-4): val is a DETERMINISTIC tail
    # slice OF the seed-0 train pairs, taken per-dataset BEFORE concatenation and
    # then interleaved. The pre-fix construction (loader(split="train", seed=1000))
    # re-permuted the FULL pool with an independent seed, so val partially overlapped
    # the reserved seed-0 test split (~20% of consumed TQA val items), and because
    # datasets were concatenated in order the 25-item val slice consumed in the train
    # loop was 100% first-dataset (TQA) — checkpoint selection never saw ARC signal.
    # The canonical offsets_mc.pt (2026-06) was trained under the PRE-fix behaviour;
    # this fix governs re-runs only and does not alter that artifact.
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    print("Phase 1: building train + val pairs (CLEAN: split='train', test reserved; "
          "val = per-dataset tail of the seed-0 train pairs, interleaved)")
    train_pairs = []
    per_ds_val = []
    n_val_per_ds = max(1, args.n_val // max(1, len(datasets)))
    for d in datasets:
        loader = DATASETS[d]
        # data_seed: 0 = canonical split (E1 holds this fixed; E2 varies it for replicates)
        ds_train = loader(args.n_train + n_val_per_ds, split="train", seed=args.data_seed)
        # tail of the SAME seed-0 train pool -> val is train-internal by construction
        per_ds_val.append(ds_train[-n_val_per_ds:])
        train_pairs.extend(ds_train[:-n_val_per_ds])
    # Interleave so any truncated val slice (the train loop consumes val_pairs[:25])
    # stays dataset-balanced instead of all-first-dataset.
    val_pairs = [p for tup in zip(*per_ds_val) for p in tup] if per_ds_val else []
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)} "
          f"({n_val_per_ds}/dataset, interleaved)")
    print(f"  NOTE: TQA test split (~140 items) and ARC test split reserved for final eval.")
    print()

    # Load model frozen. Default attn_implementation='sdpa' (safe on all
    # hardware and head_dims). Opt-in to flash_attention_2 via --attn-impl
    # for users who confirmed FA2 works on their build + head_dim. Hooks on
    # o_proj fire AFTER the attention kernel regardless of impl, so all
    # impls are hook-compatible.
    #
    # Why default sdpa (not FA2) — 2026-05-11 observation:
    # FA2's load succeeded for Gemma 4 31B (head_dim=256) but forward raised
    # `RuntimeError: FlashAttention forward only supports head dimension at most 256`
    # despite head_dim==256 being the documented limit. Some FA2 builds /
    # Blackwell kernels effectively cap below 256. SDPA is ~10-20% slower
    # than FA2 at seq=384 but works universally.
    print("Phase 2: load model + freeze")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if args.attn_impl == "flash_attention_2":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="flash_attention_2",
            )
            # Smoke forward — catches runtime-time FA2 failures (head_dim,
            # kernel build) that the load step doesn't expose.
            with torch.no_grad():
                toks_smoke = tokenizer("hello", return_tensors="pt").to("cuda")
                model(**toks_smoke, use_cache=False)
            print("  attn_implementation: flash_attention_2 (smoke OK)")
        except (ImportError, ValueError, RuntimeError) as fa2_err:
            print(f"  flash_attention_2 failed ({type(fa2_err).__name__}: {fa2_err}); "
                  f"falling back to sdpa")
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                args.base, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa",
            )
            print("  attn_implementation: sdpa (fallback)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=args.attn_impl,
        )
        print(f"  attn_implementation: {args.attn_impl}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    layers = _resolve_layers(model)
    attn_shape = _get_attention_shape(model, layers)
    print(f"  layers={len(layers)} num_q_heads={attn_shape.num_q_heads} "
          f"head_dim={attn_shape.head_dim}")

    # Detect per-layer attention dim and filter selected heads to compatible layers.
    expected_total = attn_shape.num_q_heads * attn_shape.head_dim
    layer_dims = detect_layer_dims(model, tokenizer, layers, expected_total)
    compatible_layers = {li for li, dim in layer_dims.items() if dim == expected_total}
    layer_head_pairs = [(li, hi) for li, hi in layer_head_pairs_raw if li in compatible_layers]
    skipped = [(li, hi) for li, hi in layer_head_pairs_raw if li not in compatible_layers]
    if skipped:
        print(f"  Skipped {len(skipped)} heads from incompatible layers:")
        for li, hi in skipped:
            print(f"    L{li} H{hi} (layer dim={layer_dims.get(li)}, expected {expected_total})")
    print(f"  Compatible heads to train: {len(layer_head_pairs)}")
    if not layer_head_pairs:
        raise RuntimeError("No selected heads are in compatible layers. Check head selection file.")
    print(f"  First 10 trainable:")
    for r, (li, hi) in enumerate(layer_head_pairs[:10]):
        info = next(h for h in selected_raw if h["layer"] == li and h["head"] == hi)
        print(f"    {r+1:2d}. L{li:2d} H{hi:2d}  acc={info['best_acc']:.3f}")
    print()

    # Create offsets module
    offsets = LofitOffsets(
        layer_head_pairs, head_dim=attn_shape.head_dim,
        device="cuda", dtype=torch.float32,
    )
    n_trainable = sum(p.numel() for p in offsets.parameters() if p.requires_grad)
    print(f"  trainable params (offsets): {n_trainable}")
    print()

    # P9: load + map v_inf onto the trainable pairs (on-axis penalty target). Off unless
    # --onaxis-penalty. Fail fast if nothing maps (a silent 0-mapping would make the penalty
    # a no-op and quietly produce an off-axis "on-axis" adapter).
    onaxis_v = onaxis_mask = None
    if args.onaxis_penalty:
        onaxis_v, onaxis_mask = load_vinf_for_pairs(args.vinf_path, layer_head_pairs, device="cuda")
        n_mapped = int(onaxis_mask.sum())
        print(f"  P9 on-axis penalty ON: lambda={args.onaxis_lambda}, vinf={args.vinf_path}")
        print(f"    v_inf mapped to {n_mapped}/{len(layer_head_pairs)} trainable pairs")
        if n_mapped == 0:
            raise RuntimeError(
                f"P9 on-axis penalty requested but 0/{len(layer_head_pairs)} pairs mapped to "
                f"v_inf in {args.vinf_path}. Check captured_indices vs the trained head layers."
            )
        n_unmapped = len(layer_head_pairs) - n_mapped
        if n_unmapped:
            print(f"    NOTE: {n_unmapped} unmapped pairs contribute 0 to the penalty")
        print()

    # Install hooks
    handles = install_lofit_hooks(layers, offsets, attn_shape)

    # Optimizer
    optim = torch.optim.AdamW(offsets.parameters(), lr=args.lr, weight_decay=0.0)

    # Atomic save helper. Used for both best-val-acc preservation during training
    # (out_path = best state) and the final end-of-training state (out_path with
    # .final.pt suffix). Atomic = write to .tmp then rename, so a crash mid-save
    # never leaves a corrupt file. Pattern lifted from `task_token` script
    # (commit 616e304, fix: task_token resilience).
    def _save_offsets_atomic(path: Path, val_acc: float, step_saved: int, tag: str):
        save_dict = {
            "config": {
                "base_path": args.base,
                "heads_file": args.heads_file,
                "top_k": args.top_k,
                "datasets": datasets,
                "steps": args.steps,
                "loss_type": args.loss,
                "lr": args.lr,
                "beta": args.beta,
                "gamma": args.gamma,
                "onaxis_penalty": args.onaxis_penalty,
                "onaxis_lambda": args.onaxis_lambda,
                "vinf_path": args.vinf_path if args.onaxis_penalty else None,
                "seed": args.seed,
                "data_seed": args.data_seed,
            },
            "layer_head_pairs": layer_head_pairs,
            "alpha": offsets.alpha.detach().cpu(),
            "theta": offsets.theta.detach().cpu(),
            "head_dim": attn_shape.head_dim,
            "num_q_heads": attn_shape.num_q_heads,
            "val_acc_saved": val_acc,
            "step_saved": step_saved,
            "tag": tag,
        }
        tmp = Path(str(path) + ".tmp")
        torch.save(save_dict, tmp)
        tmp.replace(path)

    # Training loop
    print("Phase 3: training")
    print()
    has_downstream = bool(args.downstream_primary)
    spillover_list = [t.strip() for t in args.downstream_spillover.split(",") if t.strip()]
    downstream_eval_every = args.downstream_eval_every if args.downstream_eval_every else args.eval_every
    # Validate tasks at startup (fail-fast, lazy-import lm_eval)
    _validate_downstream_tasks(args.downstream_primary, spillover_list)
    # Trajectory storage
    trajectory: list[TrajectoryEntry] = []
    trajectory_path = out_path.with_suffix(".trajectory.json")
    best_downstream_primary = -1.0
    first_cycle_spillover: dict[str, float] = {}  # baseline for WARN logic
    # Header
    if has_downstream:
        spill_cols = "  ".join(f"{t[:8]:>8s}" for t in spillover_list)
        spill_hdr = f"  {spill_cols}" if spillover_list else ""
        print(f"{'step':>5s}  {'loss':>9s}  {'val_loss':>9s}  {'val_acc':>8s}  "
              f"{args.downstream_primary[:8]:>8s}{spill_hdr}  {'time':>6s}")
        print("-" * (47 + 10 + 10 * len(spillover_list)))
    else:
        print(f"{'step':>5s}  {'loss':>9s}  {'val_loss':>9s}  {'val_acc':>8s}  {'time':>6s}")
        print("-" * 47)

    rng = torch.Generator().manual_seed(args.seed)
    train_log = []
    t0 = time.time()
    accum = 0
    optim.zero_grad()
    # Best-checkpoint tracking. Tiebreak (val_acc, -val_loss) so when val_acc
    # plateaus across cycles (e.g., base model already saturates the
    # contrastive task), the lower-val_loss state wins. Without this tiebreak,
    # step 0 (zero offsets) "wins" any later step with same val_acc, saving
    # essentially the base model. Observed on Path A coding adapter 2026-05-11
    # — val_acc=0.880 plateau forced best=step 1 (untrained).
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_step = 0

    n_train = len(train_pairs)

    for step in range(args.steps):
        # Sample batch
        idx = torch.randint(0, n_train, (args.batch_size,), generator=rng).tolist()

        loss_step = 0.0
        for j in idx:
            loss = task_margin_loss(
                model, tokenizer, train_pairs[j],
                seq_len_max=args.seq_len_max, device="cuda",
                loss_type=args.loss, beta=args.beta, gamma=args.gamma,
                chat_template=args.chat_template,
            )
            (loss / (args.batch_size * args.grad_accum)).backward()
            loss_step += float(loss.detach())

        accum += args.batch_size

        if accum >= args.batch_size * args.grad_accum:
            # P9: add the on-axis penalty gradient ONCE per optimizer step (it's a regularizer
            # on the params, not a per-sample term). The margin grads for this window are already
            # accumulated; this adds lam*grad(L_onaxis) before the step. No-op when penalty off.
            if args.onaxis_penalty:
                onaxis_penalty_term(offsets, onaxis_v, onaxis_mask, args.onaxis_lambda).backward()
            torch.nn.utils.clip_grad_norm_(offsets.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()
            accum = 0
            # Free fragmented activations / temporary tensors. Cheap (~5-20ms)
            # vs giving up margin for OOM under longer pairs. Lifted from
            # task_token resilience fix.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        loss_avg = loss_step / max(args.batch_size, 1)

        # Periodic eval (val cap reduced 50 -> 25 to lower per-cycle overhead;
        # eval_every default also bumped 100 -> 200; combined ~3-4x less
        # eval-time wall while still tracking val_acc trajectory)
        if (step + 1) % args.eval_every == 0 or step == 0:
            offsets.eval()
            with torch.no_grad():
                val_losses = []
                val_correct = 0
                for vp in val_pairs[: min(25, len(val_pairs))]:
                    q, c, w = vp
                    if args.chat_template:
                        prompt = _build_prompt_chat(tokenizer, q)
                        sep = ""
                    else:
                        prompt = f"Question: {q}\nAnswer:"
                        sep = " "
                    lp_c = continuation_logprob(model, tokenizer, prompt, c, args.seq_len_max, "cuda", separator=sep)
                    lp_w = continuation_logprob(model, tokenizer, prompt, w, args.seq_len_max, "cuda", separator=sep)
                    # Mirror the training objective, otherwise the logged val curve
                    # reads backwards under anti_direct (E7).
                    if args.loss == "anti_direct":
                        val_loss_one = -lp_w + args.gamma * lp_c
                    else:
                        val_loss_one = -lp_c + args.gamma * lp_w
                    val_losses.append(float(val_loss_one))
                    if lp_c.item() > lp_w.item():
                        val_correct += 1
                val_loss = sum(val_losses) / max(len(val_losses), 1)
                val_acc = val_correct / max(len(val_losses), 1)
            offsets.train()
            elapsed = time.time() - t0

            # ---- Mid-training downstream eval (opt-in, Task 6) ----
            downstream_primary_acc: float | None = None
            downstream_spillover_accs: dict[str, float | None] = {}
            step_ckpt_path: str = ""
            ran_downstream: bool = False

            if has_downstream and ((step + 1) % downstream_eval_every == 0 or step == 0):
                # 1. Save step ckpt first (so even if eval fails we have the snapshot)
                step_ckpt = out_path.with_suffix(f".step_{step+1}.pt")
                _save_offsets_atomic(step_ckpt, val_acc, step + 1, tag=f"step_{step+1}")
                step_ckpt_path = str(step_ckpt.resolve())

                # 2. Run downstream eval (lm_eval clamps limit to task size silently — Polish #6 docstring on _run_downstream_eval)
                downstream_results = _run_downstream_eval(
                    model=model,
                    tokenizer=tokenizer,
                    primary=args.downstream_primary,
                    spillover=spillover_list,
                    limit=args.downstream_limit,
                    apply_chat_template=args.downstream_apply_chat_template,
                )
                downstream_primary_acc = downstream_results.get(args.downstream_primary)
                downstream_spillover_accs = {
                    t: downstream_results.get(t) for t in spillover_list
                }

                # 3. Capture first-cycle spillover baseline for WARN logic
                if not first_cycle_spillover and any(v is not None for v in downstream_spillover_accs.values()):
                    first_cycle_spillover = {
                        t: v for t, v in downstream_spillover_accs.items() if v is not None
                    }

                # 4. Spillover WARN
                for t, acc in downstream_spillover_accs.items():
                    if acc is None or t not in first_cycle_spillover:
                        continue
                    drop_pp = (first_cycle_spillover[t] - acc) * 100.0
                    if drop_pp > args.downstream_spillover_warn_pp:
                        print(f"[downstream-eval] WARN: spillover '{t}' dropped "
                              f"{drop_pp:.1f}pp vs first cycle "
                              f"({first_cycle_spillover[t]:.3f} -> {acc:.3f})")

                ran_downstream = True

            # Lexicographic best (legacy val_pair_acc track — kept for backwards compat)
            improved_val_pair = (val_acc > best_val_acc) or (
                val_acc == best_val_acc and val_loss < best_val_loss
            )

            # Downstream best (new — primary metric drives best ckpt when feature on)
            is_best_primary = False
            if has_downstream and downstream_primary_acc is not None:
                is_best_primary = downstream_primary_acc > best_downstream_primary
                if is_best_primary:
                    best_downstream_primary = downstream_primary_acc

            # Marker logic: prefer downstream "best" marker when feature on; else val_pair
            if has_downstream:
                marker = "  *" if is_best_primary else ""
            else:
                marker = "  *" if improved_val_pair else ""

            # Print row (expanded when downstream on)
            if has_downstream:
                prim_str = f"{downstream_primary_acc:.3f}" if downstream_primary_acc is not None else "  n/a "
                spill_strs = []
                for t in spillover_list:
                    v = downstream_spillover_accs.get(t)
                    spill_strs.append(f"{v:.3f}" if v is not None else " n/a ")
                spill_col = "  ".join(f"{s:>8s}" for s in spill_strs)
                spill_part = f"  {spill_col}" if spill_strs else ""
                print(f"{step+1:>5d}  {loss_avg:>9.4f}  {val_loss:>9.4f}  {val_acc:>8.3f}  "
                      f"{prim_str:>8s}{spill_part}  {elapsed:>6.0f}s{marker}")
            else:
                print(f"{step+1:>5d}  {loss_avg:>9.4f}  {val_loss:>9.4f}  {val_acc:>8.3f}  {elapsed:>6.0f}s{marker}")

            # P9: report how on-axis the offset currently is. frac_perp -> 0 as the offset
            # moves onto v_inf; the sweep watches this fall as lambda grows (plan Task 2).
            cur_onaxis = None
            cur_frac_perp = None
            if args.onaxis_penalty:
                with torch.no_grad():
                    cur_onaxis = float(onaxis_penalty(offsets.alpha, offsets.theta, onaxis_v, onaxis_mask))
                    cur_frac_perp = frac_perp(offsets.alpha, offsets.theta, onaxis_v, onaxis_mask)
                print(f"        P9: onaxis_penalty={cur_onaxis:.5f}  frac_perp={cur_frac_perp:.3f}  "
                      f"(lambda={args.onaxis_lambda})", flush=True)

            train_log.append({
                "step": step + 1, "loss": loss_avg, "val_loss": val_loss,
                "val_acc": val_acc, "time": elapsed,
                "is_best_so_far": is_best_primary if has_downstream else improved_val_pair,
                "onaxis_penalty": cur_onaxis, "frac_perp": cur_frac_perp,
            })

            # Persist best ckpt:
            # - Feature OFF: best = val_pair_acc (current behavior, unchanged)
            # - Feature ON:  best = downstream_primary_acc (overrides val_pair)
            if has_downstream:
                if is_best_primary:
                    best_step = step + 1
                    best_val_acc = val_acc  # cosmetic; reported in summary
                    best_val_loss = val_loss
                    _save_offsets_atomic(out_path, val_acc, step + 1, tag="best_primary")
            else:
                if improved_val_pair:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_step = step + 1
                    _save_offsets_atomic(out_path, val_acc, step + 1, tag="best")

            # Append trajectory entry + atomic write (Polish item #4)
            if has_downstream and ran_downstream:
                trajectory.append(TrajectoryEntry(
                    step=step + 1,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    downstream_primary_acc=downstream_primary_acc,
                    downstream_spillover_accs=downstream_spillover_accs,
                    ckpt_path=step_ckpt_path,
                    elapsed_s=elapsed,
                    is_best_primary=is_best_primary,
                ))
                _save_trajectory_atomic(trajectory_path, trajectory, meta={
                    "primary": args.downstream_primary,
                    "spillover": spillover_list,
                    "limit": args.downstream_limit,
                    "downstream_eval_every": downstream_eval_every,
                    "apply_chat_template": args.downstream_apply_chat_template,
                    "spillover_warn_pp": args.downstream_spillover_warn_pp,
                })

    remove_handles(handles)
    print()
    print(f"Training complete. Total time: {time.time() - t0:.0f}s")
    if has_downstream:
        print(f"Best downstream-primary ({args.downstream_primary}) acc: "
              f"{best_downstream_primary:.3f} at step {best_step} -> {out_path}")
    else:
        print(f"Best val_acc: {best_val_acc:.3f} at step {best_step} -> {out_path}")
    print()

    # Fallback: if has_downstream but ALL downstream evals failed,
    # best_step is still 0 and out_path was never written. Save the
    # FINAL training state as out_path so downstream consumers don't
    # crash on missing file. Logged so this isn't silent.
    if has_downstream and best_step == 0:
        print(f"[downstream-eval] WARNING: no downstream eval succeeded; "
              f"falling back to final-state offsets for {out_path}")
        _save_offsets_atomic(out_path, val_acc, args.steps, tag="best_fallback_no_downstream")

    # Final-state save (separate from best). Useful for studying end-of-training
    # collapse / overfit pattern. Eval pipelines should consume out_path (best).
    final_path = out_path.with_suffix(".final.pt") if out_path.suffix else Path(str(out_path) + ".final")
    _save_offsets_atomic(final_path, val_acc, args.steps, tag="final")
    # P9: record the final on-axis-ness of the trained offset (the sweep's key number).
    onaxis_final = None
    frac_perp_final = None
    if args.onaxis_penalty:
        with torch.no_grad():
            onaxis_final = float(onaxis_penalty(offsets.alpha, offsets.theta, onaxis_v, onaxis_mask))
            frac_perp_final = frac_perp(offsets.alpha, offsets.theta, onaxis_v, onaxis_mask)
        print(f"P9 final: onaxis_penalty={onaxis_final:.5f}  frac_perp={frac_perp_final:.3f}  "
              f"(lambda={args.onaxis_lambda})")

    log_path.write_text(json.dumps({
        "train_log": train_log,
        "best_val_acc": best_val_acc,
        "best_step": best_step,
        "best_downstream_primary": best_downstream_primary if has_downstream else None,
        "onaxis_lambda": args.onaxis_lambda if args.onaxis_penalty else None,
        "onaxis_penalty_final": onaxis_final,
        "frac_perp_final": frac_perp_final,
    }, indent=2))
    if has_downstream:
        print(f"Saved best offsets:    {out_path} (downstream_primary="
              f"{best_downstream_primary:.3f} @ step {best_step})")
        print(f"Saved final offsets:   {final_path} (val_acc={val_acc:.3f} @ step {args.steps})")
        n_step_ckpts = sum(1 for e in trajectory)
        print(f"Saved {n_step_ckpts} step ckpts: {out_path.parent}/{out_path.stem}.step_*.pt")
        print(f"Saved trajectory:      {trajectory_path}")
    else:
        print(f"Saved best offsets:  {out_path} (val_acc={best_val_acc:.3f} @ step {best_step})")
        print(f"Saved final offsets: {final_path} (val_acc={val_acc:.3f} @ step {args.steps})")
    print(f"Saved log:           {log_path}")


if __name__ == "__main__":
    main()
