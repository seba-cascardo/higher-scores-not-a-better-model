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
        --base /home/seba_/models/gemma4-e2b \\
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


def build_tqa_pairs(n: int, split: str = "train", train_frac: float = TRAIN_FRAC, seed: int = 0):
    """TQA mc1 contrastive pairs. TQA only has 'validation' split natively;
    we deterministically split it into our own train/test using seed-0 perm.

    split='train': first floor(train_frac * n_valid) items
    split='test':  remaining items (disjoint from train by construction)

    n caps the returned size after split selection.
    """
    print(f"  TQA: loading split={split} (target n={n})...", flush=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()

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
        all_valid.append((ex["question"], choices[cidx], choices[wrong[0]]))

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

    def get_layer_offsets(self, layer_idx: int) -> dict[int, torch.Tensor]:
        """Return {head_idx: alpha * theta} for all trainable heads at this layer."""
        if layer_idx not in self.by_layer:
            return {}
        out = {}
        for hi in self.by_layer[layer_idx]:
            idx = self.pair_to_idx[(layer_idx, hi)]
            out[hi] = self.alpha[idx] * self.theta[idx]  # (head_dim,)
        return out


# -- Hook installation -------------------------------------------------------
def install_lofit_hooks(layers, offsets: LofitOffsets, attn_shape: AttentionShape):
    """Register pre-hooks on attn.o_proj for each layer with selected heads.

    Hook adds `α_h · θ_h` to the per-head slice of o_proj's input.
    Returns list of handles for cleanup.

    Defensive shape check: Gemma 4 has alternating sliding/global attention
    where global layers have 2× the q-dim. If a selected head comes from a
    non-matching layer, the reshape would fail. We raise a clear error.
    """
    expected_total = attn_shape.num_q_heads * attn_shape.head_dim
    handles = []
    for layer_idx in offsets.by_layer:
        layer = layers[layer_idx]
        o_proj = layer.self_attn.o_proj

        def make_hook(li):
            def hook(_module, inputs):
                concat = inputs[0]  # (B, T, total)
                B, T, total = concat.shape
                if total != expected_total:
                    raise RuntimeError(
                        f"Layer {li}: o_proj input dim {total} != expected "
                        f"{expected_total} ({attn_shape.num_q_heads} × "
                        f"{attn_shape.head_dim}). This layer has different "
                        f"attention structure (likely global vs sliding). "
                        f"Selected head file should not include heads from this layer."
                    )
                # Build offset tensor; needs grad to flow to theta/alpha.
                offset_view = torch.zeros(
                    B, T, attn_shape.num_q_heads, attn_shape.head_dim,
                    dtype=concat.dtype, device=concat.device,
                )
                for hi, atheta in offsets.get_layer_offsets(li).items():
                    offset_view[:, :, hi, :] = atheta.to(dtype=concat.dtype)
                offset = offset_view.view(B, T, total)
                modified = concat + offset
                return (modified,) + inputs[1:]
            return hook

        handles.append(o_proj.register_forward_pre_hook(make_hook(layer_idx)))
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
    elif loss_type == "dpo":
        margin = beta * (lp_c - lp_w)
        return -F.logsigmoid(margin)
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")


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
    ap.add_argument("--loss", default="direct", choices=["direct", "dpo"])
    ap.add_argument("--seq-len-max", type=int, default=384)
    ap.add_argument("--eval-every", type=int, default=200,
                    help="Run val every N steps. Bumped to 200 default 2026-05-11 "
                         "to reduce eval overhead on Gemma 4 31B BF16 / 96GB Blackwell; "
                         "val_pairs (cap 25 per cycle) cost ~10-20s on 31B and we don't "
                         "need finer monitoring resolution. Set 100 for fine-grained probe.")
    ap.add_argument("--attn-impl", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="Attention implementation. Default 'sdpa' is safe on all hardware "
                         "and head_dims. 'flash_attention_2' may be ~10-25% faster on long "
                         "seq lengths BUT fails at runtime on Gemma 4 (head_dim=256) on some "
                         "FA2 builds even when the load succeeds — observed 2026-05-11 on "
                         "Blackwell. Use 'eager' only for debugging hook interactions.")
    ap.add_argument("--chat-template", action="store_true",
                    help="Use chat template wrap for IT models. REQUIRED for chat-aligned "
                         "deploy: V7 trained on raw text Q:/A: does not transfer to chat-template "
                         "inference. Builds prompts via apply_chat_template + add_generation_prompt. "
                         "See feedback_v7_chat_template_train_deploy_gap.md.")
    ap.add_argument("--out", required=True)
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
    print()

    # Load heads
    heads_data = json.loads(Path(args.heads_file).read_text())
    ranked = heads_data["ranked_top_k"]
    selected_raw = ranked[: args.top_k]
    layer_head_pairs_raw = [(h["layer"], h["head"]) for h in selected_raw]
    print(f"  Loaded top-{args.top_k} heads from file (will filter to compatible layers)")
    print()

    # Build datasets — clean splits, train/test disjoint by construction.
    # Both train and val use split="train" (val is just a held-out shuffle within
    # the train pool for monitoring loss; the FINAL eval uses split="test" which
    # this training never sees).
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    print("Phase 1: building train + val pairs (CLEAN: split='train', test reserved)")
    train_pairs = []
    val_pairs = []
    for d in datasets:
        loader = DATASETS[d]
        train_pairs.extend(loader(args.n_train, split="train", seed=0))
        val_pairs.extend(loader(args.n_val, split="train", seed=1000))
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)}")
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
    print(f"{'step':>5s}  {'loss':>9s}  {'val_loss':>9s}  {'val_acc':>8s}  {'time':>6s}")
    print("-" * 47)

    rng = torch.Generator().manual_seed(0)
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
                    val_loss_one = -lp_c + args.gamma * lp_w
                    val_losses.append(float(val_loss_one))
                    if lp_c.item() > lp_w.item():
                        val_correct += 1
                val_loss = sum(val_losses) / max(len(val_losses), 1)
                val_acc = val_correct / max(len(val_losses), 1)
            offsets.train()
            elapsed = time.time() - t0
            # Lexicographic best: prefer higher val_acc; if tied, prefer lower
            # val_loss. Ensures plateau cycles still update toward the
            # better-trained state.
            improved = (val_acc > best_val_acc) or (
                val_acc == best_val_acc and val_loss < best_val_loss
            )
            marker = "  *" if improved else ""
            print(f"{step+1:>5d}  {loss_avg:>9.4f}  {val_loss:>9.4f}  {val_acc:>8.3f}  {elapsed:>6.0f}s{marker}")
            train_log.append({
                "step": step + 1, "loss": loss_avg, "val_loss": val_loss,
                "val_acc": val_acc, "time": elapsed,
                "is_best_so_far": improved,
            })
            # Best-checkpoint persistence: every val cycle, if (val_acc, -val_loss)
            # improved we atomically save to out_path. If training OOMs / crashes
            # later, out_path still holds the best state seen. Eval scripts and
            # bake read out_path so this is the deploy-worthy artifact.
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_step = step + 1
                _save_offsets_atomic(out_path, val_acc, step + 1, tag="best")

    remove_handles(handles)
    print()
    print(f"Training complete. Total time: {time.time() - t0:.0f}s")
    print(f"Best val_acc: {best_val_acc:.3f} at step {best_step} -> {out_path}")
    print()

    # Final-state save (separate from best). Useful for studying end-of-training
    # collapse / overfit pattern. Eval pipelines should consume out_path (best).
    final_path = out_path.with_suffix(".final.pt") if out_path.suffix else Path(str(out_path) + ".final")
    _save_offsets_atomic(final_path, val_acc, args.steps, tag="final")
    log_path.write_text(json.dumps({
        "train_log": train_log,
        "best_val_acc": best_val_acc,
        "best_step": best_step,
    }, indent=2))
    print(f"Saved best offsets:  {out_path} (val_acc={best_val_acc:.3f} @ step {best_step})")
    print(f"Saved final offsets: {final_path} (val_acc={val_acc:.3f} @ step {args.steps})")
    print(f"Saved log:           {log_path}")


if __name__ == "__main__":
    main()
