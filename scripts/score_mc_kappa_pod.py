"""GATE-2 head-to-head — KAPPA as the MANDATORY ON-AXIS FOIL.  GPU / pod.

================================================================================
WHAT THIS IS
================================================================================
A reimplementation of **KAPPA** (Park, Pyun, Jo; "Bridging the Knowledge-Prediction
Gap"; ICML 2026; arXiv:2509.23782) as a baseline arm for GATE 2 of the off-axis
investigation (docs/superpowers/specs/2026-06-23-offaxis-mechanism-battery-and-harness-spec.md).

KAPPA is the METHODOLOGICAL OPPOSITE of our trained off-axis offset:
  - It is a CLOSED-FORM, inference-time, per-input, minimal-L2 (min-||h'-h||_2)
    affine PROJECTION that moves the hidden state WITHIN a "prediction subspace"
    span(W_pred) so its prediction-coordinates EQUAL its knowledge-coordinates
    (their Eq. 3). It leaves the orthogonal complement of span(W_pred) unchanged.
  - It is explicitly ON-AXIS (toward a probe-defined knowledge direction).
  - It is NOT trained: the only fit is two linear probes (W_know, W_pred).
  - Reported effect: modest, monotone, NON-destructive (AGR up to +22pp; ACC
    +1.5-16.9pp on high-gap tasks; "only modest changes" on MMLU/ARC; never
    damages MMLU/ARC; +2.5pp single-task free-form with a GSM8k DROP).

We use it as the on-axis reference in the GATE-2 head-to-head: same MC scoring
protocol (chat-template, teacher-forced option-text continuation, acc_norm,
same items/order/n) as our off-axis offset, on ARC (+TQA), so GATE 2 can compare
off-axis V7-mc vs KAPPA vs DC-PMI vs DoLa.

GATE-2 KILL-RULE (pre-reg, from the battery spec §1):
  if off-axis recovery on PMI-residual ARC items is within bootstrap CI of
  KAPPA/DoLa AND with >= their collateral -> no practical novelty.
GATE-2 WIN: off-axis recovers PMI-robust items KAPPA (on-axis) cannot.
This script produces the base / KAPPA / off-axis arms' acc / acc_norm and per-item
per-option log-probs so the local head-to-head + collateral analysis can be assembled.
The doc_ids align with the COMMITTED CORPUS (runs/correctness_probe_postgen/
corpus_*.jsonl). They share only the doc_id INDEX with the lm-eval d_*.json off-axis
run — NOT directly splice-comparable numbers (d_*.json used a different thinking-mode
scaffold + no-leading-space continuation; see the FAITHFUL-vs-RECONSTRUCTED note on
scoring protocol). The head-to-head must use THIS script's internally re-scored arms.

================================================================================
*** FAITHFUL-vs-RECONSTRUCTED DESIGN POINTS (READ BEFORE TRUSTING NUMBERS) ***
================================================================================
This is a reimplementation FROM THE PAPER DESCRIPTION (paper body v4, read via the
deep-research workflow `_wf_kappa.json`; the authors' repo holi-lab/KAPPA was a stub
at audit time, so NO code-level confirmation was possible). Each design point is
tagged so a reviewer can audit exactly where we are faithful to KAPPA and where we
reconstructed/deviated.

[FAITHFUL — Eq. 3, the closed-form projection]
  KAPPA's central intervention (their Eq. 3, as transcribed from the body):
      h' = h + W_pred (W_pred^T W_pred)^{-1} ( W_know^T h  -  W_pred^T h )
  This is the unique minimal-L2 update that (a) lives entirely in span(W_pred),
  and (b) makes the prediction-coordinates of h' (W_pred^T h') equal the
  knowledge-coordinates (W_know^T h). We implement EXACTLY this algebra
  (`kappa_project`), including the (W_pred^T W_pred)^{-1} Gram inverse and the
  optional sharpening scalars (alpha on the knowledge target, beta on the pred
  term) the paper mentions. With alpha=beta=1 and W_pred orthonormal it reduces
  to h + W_pred(W_know^T h - W_pred^T h). VERIFIED algebraically:
  W_pred^T h' = W_know^T h  (prediction coords replaced by knowledge coords) and
  (I-P_pred) h' = (I-P_pred) h  (orthogonal complement untouched).

[FAITHFUL — applied at a few intermediate layers]
  KAPPA is applied at ~1-6 intermediate layers. We apply at user-chosen --layers
  (default the scoring band L30/40/55 used elsewhere in this project). KAPPA(k)
  in the paper = applied at k layers; --layers length is our k.

[FAITHFUL — closed-form, inference-time, no gradient training]
  No offset is trained. The only fitted objects are the two probes (below).

[RECONSTRUCTED — subspace SPACE: per-head o_proj-input, not residual d_model]
  KAPPA in the paper operates on the RESIDUAL stream (d_model) with full probe
  matrices W in R^{d_model x k}. We instead operate PER-HEAD on the o_proj-INPUT
  slice (head_dim=256 per head), the SAME space where our W_know foil and theta_mc
  live, and install the projection as a forward_pre_hook on layer.self_attn.o_proj
  (the exact hook site of install_lofit_hooks). RATIONALE (load-bearing for a fair
  GATE-2): the apples-to-apples requirement of this project is same-distribution
  AND same-space (memory: forward_filter_same_distribution_same_space). Building
  KAPPA in the SAME per-head o_proj-input space as the off-axis offset makes the
  head-to-head fair; a residual-d_model KAPPA would be a different-space confound.
  We block-apply Eq. 3 independently within each modulated head's 256-dim slice.
  DEVIATION FROM PAPER: paper = one residual-stream projection; ours = a direct
  sum of per-head projections over the adapter's (layer,head) set. The algebra of
  Eq. 3 is faithful WITHIN each head block; the global operator is the block-diagonal
  composition. A reviewer wanting the literal residual-space KAPPA should note this.

[RECONSTRUCTED — knowledge subspace W_know source]
  Paper W_know = a k-class linear probe predicting the GROUND-TRUTH answer from the
  residual stream. We do not have that exact probe per head. We build W_know per head
  as the Fisher-LDA correctness direction from BASE activations predicting
  ground-truth correctness — identical construction to scripts/make_wknow_offset.py
  (the project's canonical on-axis foil, 6.4% ARC recovery):
      W_know_head = unit( Sigma_w^{-1} (mu_inf_correct - mu_inf_wrong) )
  with Sigma_w = cov_inf[(layer,head)] (pooled within-class scatter; ridge-shrunk).
  This is on-axis-toward-knowledge (matches KAPPA's intent) AND same-space/same-
  distribution as our foil. Default k_know=1 (rank-1 knowledge direction per head);
  --k-know>1 would need a multi-direction knowledge probe we do not currently have,
  so k_know=1 is the only fully-backed setting. DEVIATION: ground-truth-correctness
  LDA direction, not the paper's k-class option probe; same on-axis intent, different
  estimator. Built ONLY from base activations + ground-truth labels (never adapter
  behavior) -> non-circular, like W_know.

[RECONSTRUCTED — prediction subspace W_pred source]
  Paper W_pred = a probe predicting the MODEL'S OWN greedy-decoded option. We do not
  have that probe. We reconstruct W_pred per head with a documented proxy, selectable
  via --wpred-source:
    * "pc1"   (DEFAULT): the top principal component(s) of cov_inf[(layer,head)] —
              the direction the head's last-token state varies along most. This is the
              natural "prediction-coordinate" axis (what the model's readout reads),
              and it is what Eq. 3 projects WITHIN. Reconstructed, NOT the paper's
              option-probe; documented proxy.
    * "vinf": the mass-mean correctness direction v_inf (unit). A cruder proxy where
              prediction and knowledge axes nearly coincide (then KAPPA ~= a small
              push toward knowledge along v_inf). Provided for sensitivity.
    * "wknow": degenerate self-test — W_pred == W_know. Eq. 3 then makes pred coords
              equal knowledge coords that are already W_know coords -> near-identity.
              Sanity arm only (KAPPA should do ~nothing); NOT a real KAPPA setting.
  k_pred = --k-pred (default 1). DEVIATION: PC1 / v_inf are reconstructed stand-ins
  for the paper's greedy-option probe. We surface --wpred-source so the dependence on
  this reconstruction is auditable; the headline KAPPA arm is pc1.

[DEVIATION / NOTE — sharpening hyperparams alpha,beta]
  The paper mentions optional sharpening; exact defaults were not transcribed. We
  default alpha=beta=1.0 (the clean min-L2 form). --kappa-alpha/--kappa-beta expose
  them for sensitivity but the headline run uses 1.0/1.0.

[FAITHFUL — scoring protocol parity with the off-axis arm]
  Context = user turn, chat-template wrapped, add_generation_prompt=True; continuation
  = " " + option_text; score = teacher-forced sum log-prob over continuation tokens;
  acc_norm = argmax of length-(char)-normalized logprob == gold. IDENTICAL to
  scripts/score_mc_premise_pod.py and lm-eval multiple_choice. The off-axis arm is
  RE-SCORED HERE under this exact protocol (it is NOT read from d_*.json), so the
  base/KAPPA/off-axis head-to-head is internally consistent by construction.
  [DOC-ID ALIGNMENT — NOT 1:1 with d_*.json NUMBERS] Items/order/n are read from the
  COMMITTED CORPUS (CORPUS[task] = runs/correctness_probe_postgen/corpus_*.jsonl).
  The d_null/d_mc/d_vinf.json files are lm-eval HARNESS outputs (top-level results/
  samples/configs/...; per-doc under samples[task][i]) whose context ends with a
  thinking-mode scaffold ('...Answer:' then '<|channel>thought<channel|>') and uses a
  NO-leading-space continuation — a DIFFERENT protocol from this script. Therefore the
  d_*.json acc / per-option logprob VALUES are NOT splice-comparable to this script's
  numbers. Only the doc_id INDEX is shared. If d_*.json per-item logprobs are wanted for
  a PMI-residual splice, recompute the PMI-residual item selection under THIS script's
  protocol; do NOT mix d_*.json logprob values into this head-to-head.

SANITY GATES (printed; abort cheaply if violated):
  (1) SCAFFOLD validity (IMPLEMENTED, reviewer fix 2): the FULL off-axis arm's
      conditional acc_norm lift must reproduce the canonical ~+0.35 ARC / +0.33 TQA
      (CANONICAL_OFFAXIS_COND_LIFT, |Δ| <= SANITY_LIFT_EPS). This single check
      validates BOTH the scoring scaffold (chat-template / continuation tokenization;
      a broken scaffold would make every number silently wrong) AND the off-axis
      install. It supersedes a raw "base acc_norm == canonical base level" check
      because the d_*.json base level is a DIFFERENT (lm-eval thinking-mode) scaffold
      and is not splice-comparable to this script's base. Skipped only if
      --skip-offaxis-arm (then base level is NOT cross-checked — printed as such).
  (2) KAPPA must be on-axis NON-destructive: KAPPA acc_norm should be >= base - eps
      (KAPPA never damages ARC in the paper). A large KAPPA DROP => bug in Eq. 3 or
      a degenerate subspace; inspect before trusting.
  (3) probe diagnostics: cos(W_know, theta_mc) (~0 expected, off-axis), cos(W_pred,
      W_know), per-head Gram condition number.

HEAD-SET PARITY (reviewer fix 1): the GATE-2 headline is Δ(offaxis_restricted −
  KAPPA), where offaxis_restricted zeroes the trained offset on every (layer,head)
  KAPPA did NOT build, so both arms modulate the SAME cells. The FULL off-axis arm
  (all adapter heads) is reported as CONTEXT only — never as the headline (a 48-head
  vs ~10-head comparison is a head-count confound, not a mechanism result). With the
  DEFAULT --layers (None = full adapter layer set), KAPPA covers every adapter cell,
  so offaxis_restricted == the full arm and parity holds without restriction.

================================================================================
Usage (pod, GPU; fresh shell each block -> cd + set $BASE):
  cd /workspace/MSAP && pwd
  BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' \
        2>/dev/null | grep -v '/.locks/' | head -1)")
  python scripts/score_mc_kappa_pod.py \
      --base "$BASE" \
      --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --func runs/functional_directions.pt \
      --task arc_challenge --limit 400 --apply-chat-template \
      --wpred-source pc1 \
      --out runs/gate2/kappa_arc.json
  # --layers omitted -> defaults to the FULL adapter layer set (head-set parity vs
  #   KAPPA). Pass --layers L1 L2 ... only for a deliberate KAPPA(k) layer ablation.
  # then truthfulqa_mc1: --task truthfulqa_mc1 --out runs/gate2/kappa_tqa.json
  # local: assemble the GATE-2 head-to-head from kappa_*.json using the offaxis_restricted
  #   (NOT offaxis_full) arm vs kappa; recompute PMI-residual selection under THIS protocol.

Compile-check (no model):  python -c "import ast,sys; ast.parse(open('scripts/score_mc_kappa_pod.py').read())"
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

# ---- Scoring premise (parity with score_mc_premise_pod.py) ------------------
TQA_PRIMER = (
    "Q: What is human life expectancy in the United States?\n"
    "A: Human life expectancy in the United States is 78 years.\n\n"
    "Q: Who was president of the United States in 1955?\n"
    "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
    "Q: Which party did he belong to?\n"
    "A: He belonged to the Republican Party.\n\n"
    "Q: What is the square root of banana?\n"
    "A: I have no comment.\n\n"
    "Q: How does a telescope work?\n"
    "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
    "Q: Where were the 1992 Olympics held?\n"
    "A: The 1992 Olympics were held in Barcelona, Spain.\n\n"
)
PREMISE = {
    "arc_challenge":  {"prefix": "",         "cond": "Question: {q}\nAnswer:"},
    "truthfulqa_mc1": {"prefix": TQA_PRIMER, "cond": "Q: {q}\nA:"},
}
CORPUS = {
    "arc_challenge": "runs/correctness_probe_postgen/corpus_arc_challenge.jsonl",
    "truthfulqa_mc1": "runs/correctness_probe_postgen/corpus_truthfulqa_mc1.jsonl",
}

# Canonical off-axis CONDITIONAL acc_norm lift under THIS script's protocol
# (chat-template + leading-space option-text continuation). These are the +0.35
# ARC / +0.33 TQA the sibling score_mc_premise_pod.py gates against. They are the
# scaffold-validity anchor for sanity gate (1): if the FULL off-axis arm here does
# NOT reproduce ~this lift, the scoring scaffold (chat-template / continuation
# tokenization) is broken and every downstream number is silently wrong. NOTE:
# these are NOT the lm-eval d_*.json base levels (different thinking-mode scaffold,
# no-leading-space continuation) — see fix-3 docstring note. eps = tolerance.
CANONICAL_OFFAXIS_COND_LIFT = {"arc_challenge": 0.35, "truthfulqa_mc1": 0.33}
SANITY_LIFT_EPS = 0.15  # |observed - canonical| > eps => scaffold WARNING


def load_items(task: str, limit: int):
    """(doc_id, question, options, gold_idx) from the committed post-gen corpus."""
    items = []
    with open(CORPUS[task], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            opts = list(r["choices"])
            gold = list(r["choice_labels"]).index(r["gold_letter"])
            items.append((int(r["doc_id"]), r["problem"], opts, gold))
            if limit and len(items) >= limit:
                break
    return items


def build_context(tokenizer, premise_str: str, apply_chat_template: bool) -> str:
    if apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": premise_str}],
            tokenize=False, add_generation_prompt=True)
    return premise_str


@torch.no_grad()
def cont_logprob(model, tokenizer, context_str: str, continuation: str) -> float:
    """Teacher-forced sum log-prob of `continuation` given `context_str`.
    Mirrors lm-eval loglik (and score_mc_premise_pod.py). add_special_tokens=False
    because the chat template already adds BOS."""
    ctx = tokenizer(context_str, add_special_tokens=False).input_ids
    full = tokenizer(context_str + continuation, add_special_tokens=False).input_ids
    n_ctx = len(ctx)
    if len(full) <= n_ctx:
        return 0.0
    ids = torch.tensor([full], device=model.device)
    logits = model(ids, use_cache=False).logits[0]            # [T, vocab]
    lp = torch.log_softmax(logits[n_ctx - 1:-1].float(), dim=-1)
    tgt = torch.tensor(full[n_ctx:], device=lp.device)
    return float(lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item())


# ============================================================================
# KAPPA closed-form projection (Eq. 3) — FAITHFUL algebra, per-head block form.
# ============================================================================
def kappa_project(x, W_know, W_pred, alpha=1.0, beta=0.0, ridge=1e-6):
    """Apply KAPPA Eq. 3 to the last-dim coordinates of x (per head block).

    Eq. 3/4 (paper form — UNIFIED 2026-06-24 with score_mc_kappa_faithful_pod.py so C3
    compares the SAME Eq.3 in both spaces; audit A2):
        target = alpha * W_know^T h  +  beta * sign(W_know^T h)      # beta = SIGN sharpen
        h' = h + W_pred (W_pred^T W_pred)^{-1} ( target  -  W_pred^T h )

    The `- W_pred^T h` is STRUCTURAL (always present), NOT beta — beta is the paper's
    additive sign-sharpening term (Eq.4), default 0 = pure Eq.3. (The old default
    `alpha*ck - beta*cp` with beta=1 was numerically IDENTICAL to this at beta=0, so the
    already-run KS sweep is unchanged; this only fixes the beta SEMANTICS.) This is the
    minimal-L2 update living in span(W_pred); the orthogonal complement is untouched. With
    alpha=1,beta=0 and W_pred orthonormal it reduces to h + W_pred ( W_know^T h - W_pred^T h ).

    Args:
      x      : [..., d]  (d = head_dim; the o_proj-input slice for one head)
      W_know : [d, k_know]  knowledge subspace basis (cols)
      W_pred : [d, k_pred]  prediction subspace basis (cols)
      alpha  : knowledge-coordinate amplification (default 1.0 = Eq.3)
      beta   : additive SIGN-sharpening term (default 0.0 = pure Eq.3; paper Eq.4)
      ridge  : tiny jitter on the Gram before inverse (numerical safety)

    Returns x' same shape/dtype/device as x.

    [FAITHFUL] This is Eq. 3 exactly. [RECONSTRUCTED] The per-head BLOCK
    application (rather than one residual-space projection) is the documented
    space deviation (see module docstring).
    """
    dt = x.dtype
    xf = x.float()                                   # [..., d]
    Wk = W_know.float()                              # [d, kk]
    Wp = W_pred.float()                              # [d, kp]
    # coordinates
    ck = torch.matmul(xf, Wk)                        # [..., kk]  = W_know^T h (rows)
    cp = torch.matmul(xf, Wp)                        # [..., kp]  = W_pred^T h
    # target in pred coordinates: alpha*knowledge + beta*sign(knowledge); minus the
    # structural pred coords. (the paper aligns pred coords TO knowledge coords; same count.)
    if ck.shape[-1] != cp.shape[-1]:
        raise ValueError(
            f"kappa_project: k_know ({ck.shape[-1]}) != k_pred ({cp.shape[-1]}); "
            f"Eq. 3 aligns prediction coords to knowledge coords and needs equal rank.")
    target = alpha * ck + (beta * torch.sign(ck) if beta != 0.0 else 0.0)
    delta_coords = target - cp                      # [..., k]  (- cp is structural, not beta)
    # (W_pred^T W_pred)^{-1}  (Gram inverse; ridge for safety)
    gram = torch.matmul(Wp.t(), Wp)                 # [kp, kp]
    gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    gram_inv = torch.linalg.inv(gram)               # [kp, kp]
    # update = W_pred @ gram_inv @ delta_coords
    upd = torch.matmul(torch.matmul(delta_coords, gram_inv), Wp.t())  # [..., d]
    return (xf + upd).to(dt)


def install_kappa_hooks(layers, attn_shape, head_bases, kappa_alpha, kappa_beta):
    """Install forward_pre_hooks on layer.self_attn.o_proj that apply KAPPA Eq. 3
    per-head to the o_proj-input slice (SAME hook site as install_lofit_hooks).

    head_bases: dict[layer_idx] -> list of (head_idx, W_know[d,kk], W_pred[d,kp], theta_mc[d]).
        (4th element added by Cambio 1 for the geometry-only decomposition; unpacked
        here with `*_` so it is ignored by the KAPPA scoring path.)
    For each modulated head h, the [.., head_dim] slice
        concat[..., hi*head_dim : (hi+1)*head_dim]
    is replaced by kappa_project(slice, W_know, W_pred). Heads NOT in head_bases
    pass through unchanged (block-diagonal identity off the modulated heads).

    Returns hook handles. Mirrors the device/dtype handling of install_lofit_hooks.
    """
    head_dim = attn_shape.head_dim
    num_q_heads = attn_shape.num_q_heads
    expected_total = num_q_heads * head_dim
    handles = []
    for layer_idx, items in head_bases.items():
        layer = layers[layer_idx]
        if not hasattr(layer, "self_attn"):
            print(f"  WARN: layer {layer_idx} has no self_attn, skipping", flush=True)
            continue
        o_proj = layer.self_attn.o_proj
        dev, dtp = o_proj.weight.device, o_proj.weight.dtype
        prepared = [(hi, Wk.to(device=dev, dtype=torch.float32),
                     Wp.to(device=dev, dtype=torch.float32))
                    for (hi, Wk, Wp, *_) in items]

        def make_hook(li, prepared, _ad=kappa_alpha, _bd=kappa_beta):
            def hook(_module, inputs):
                concat = inputs[0]
                B, T, total = concat.shape
                if total != expected_total:
                    raise RuntimeError(
                        f"L{li}: o_proj input dim {total} != expected {expected_total}")
                out = concat.clone()
                for hi, Wk, Wp in prepared:
                    sl = out[:, :, hi * head_dim:(hi + 1) * head_dim]   # [B,T,d]
                    out[:, :, hi * head_dim:(hi + 1) * head_dim] = kappa_project(
                        sl, Wk, Wp, alpha=_ad, beta=_bd)
                return (out,) + inputs[1:]
            return hook

        handles.append(o_proj.register_forward_pre_hook(make_hook(layer_idx, prepared)))
    return handles


def restrict_offaxis_to_kappa_cells(lhp, alpha, theta, head_bases):
    """Zero the off-axis offset on every (layer,head) NOT built by KAPPA.

    GATE-2 head-set parity (reviewer fix 1): the headline Δ must compare the
    off-axis offset and KAPPA over the SAME modulated cells. The full V7-mc
    adapter touches all (layer,head) in `lhp`; KAPPA only the subset in
    `head_bases` (its keys are layers, each value lists the built head indices).
    This returns (alpha_r, theta_r) — clones of alpha/theta with all rows whose
    (layer,head) is not in the KAPPA-built set set to zero — so installing them
    via install_lofit_hooks(layers, alpha_r, theta_r, lhp, ...) modulates EXACTLY
    the KAPPA cells (same hook codepath, same lhp order, identity off the KAPPA
    cells). Returns (alpha_r, theta_r, n_offaxis_restricted_heads, kappa_cells).
    """
    kappa_cells = {(int(li), int(hi)) for li, items in head_bases.items()
                   for (hi, _Wk, _Wp, *_) in items}
    alpha_r = alpha.clone()
    theta_r = theta.clone()
    n_kept = 0
    for idx, (li, hi) in enumerate(lhp):
        if (int(li), int(hi)) in kappa_cells:
            n_kept += 1
        else:
            alpha_r[idx] = 0
            theta_r[idx] = 0
    return alpha_r, theta_r, n_kept, kappa_cells


# ---- Subspace construction (W_know reconstructed; W_pred reconstructed) -----
def _lda_direction(cov, v, ridge_frac):
    """Sigma_w^{-1} v (Fisher-LDA), unit-normalized. Mirrors make_wknow_offset.py."""
    cov = cov.to(torch.float64)
    ridge = ridge_frac * torch.diag(cov).mean()
    cov_r = cov + ridge * torch.eye(cov.shape[0], dtype=torch.float64)
    w = torch.linalg.solve(cov_r, v.to(torch.float64))
    n = float(torch.linalg.norm(w))
    return (w / n).float() if n > 1e-12 else None


def _top_pcs(cov, k, ridge_frac):
    """Top-k eigenvectors of cov (descending eigenvalue). [d, k], orthonormal."""
    cov = cov.to(torch.float64)
    ridge = ridge_frac * torch.diag(cov).mean()
    cov_r = cov + ridge * torch.eye(cov.shape[0], dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(cov_r)            # ascending
    idx = torch.argsort(evals, descending=True)[:k]
    return evecs[:, idx].float()                        # [d, k]


def build_head_bases(pairs, fd, k_know, k_pred, wpred_source, ridge_frac,
                     only_layers):
    """Build per-(layer,head) W_know [d,kk] and W_pred [d,kp].

    Returns (head_bases dict, diagnostics dict).
    Only heads on `only_layers` (the --layers KAPPA application set) AND present in
    the E2 grid (captured_indices) AND eligible are included.
    """
    mu_c = fd["mu_inf_correct"]                          # [n_lay, nH, D]
    mu_w = fd["mu_inf_wrong"]
    v_inf = fd["v_inf"]
    theta_mc = fd["theta_mc"]
    cov_inf = fd["cov_inf"]                              # dict (layer,head)->[D,D]
    captured = [int(x) for x in fd["captured_indices"]]
    cap_index = {li: ai for ai, li in enumerate(captured)}
    head_dim = int(fd.get("head_dim", v_inf.shape[-1]))

    head_bases: dict[int, list] = {}
    n_built = n_skip_layer = n_skip_nocov = n_skip_null = 0
    cos_wknow_theta, cos_wpred_wknow, gram_conds = [], [], []

    for (li, hi) in pairs:
        if only_layers and li not in only_layers:
            continue
        if li not in cap_index:
            n_skip_layer += 1
            continue
        ai = cap_index[li]
        cov = cov_inf.get((li, hi))
        if cov is None:
            n_skip_nocov += 1
            continue
        # --- W_know: Fisher-LDA correctness direction (k_know=1 backed) ---
        v = (mu_c[ai, hi] - mu_w[ai, hi])
        wk = _lda_direction(cov, v, ridge_frac)         # [d] unit, or None
        if wk is None:
            n_skip_null += 1
            continue
        W_know = wk.unsqueeze(1)                          # [d, 1]
        if k_know > 1:
            # Augment with top PCs of cov as extra knowledge directions (DEVIATION:
            # no multi-class knowledge probe available; PC augmentation is a stand-in).
            extra = _top_pcs(cov, k_know - 1, ridge_frac)
            W_know = torch.cat([W_know, extra], dim=1)    # [d, k_know]

        # --- W_pred: reconstructed prediction subspace ---
        if wpred_source == "pc1":
            W_pred = _top_pcs(cov, k_pred, ridge_frac)    # [d, k_pred]
        elif wpred_source == "vinf":
            vv = v_inf[ai, hi]
            nn = float(torch.linalg.norm(vv))
            if nn < 1e-12:
                n_skip_null += 1
                continue
            W_pred = (vv / nn).unsqueeze(1)
            if k_pred > 1:
                W_pred = torch.cat(
                    [W_pred, _top_pcs(cov, k_pred - 1, ridge_frac)], dim=1)
        elif wpred_source == "wknow":
            W_pred = W_know.clone()
        else:
            raise ValueError(f"unknown wpred-source {wpred_source}")

        # Eq. 3 coordinate subtraction needs matching rank.
        if W_know.shape[1] != W_pred.shape[1]:
            kc = min(W_know.shape[1], W_pred.shape[1])
            W_know, W_pred = W_know[:, :kc], W_pred[:, :kc]

        # 4th element: the trained off-axis offset theta_mc for THIS head (NOT unit;
        # alpha*direction in o_proj-input space). Exposed so the geometry-only
        # decomposition (scripts/decompose_theta_mc_kappa_subspaces.py) can reuse the
        # EXACT same W_know/W_pred this builds, byte-identical to the GATE-2 foil.
        # Earlier call-sites (install_kappa_hooks / restrict_offaxis_to_kappa_cells)
        # unpack with `*_` so the extra element is backward-compatible.
        head_bases.setdefault(li, []).append(
            (hi, W_know, W_pred, theta_mc[ai, hi].detach().clone()))
        n_built += 1

        # diagnostics
        th = theta_mc[ai, hi]
        tn = float(torch.linalg.norm(th))
        if tn > 1e-9:
            cos_wknow_theta.append(float(W_know[:, 0] @ (th / tn)))
        cos_wpred_wknow.append(
            float(abs(W_pred[:, 0] @ W_know[:, 0]) /
                  (float(torch.linalg.norm(W_pred[:, 0])) *
                   float(torch.linalg.norm(W_know[:, 0])) + 1e-12)))
        gram = (W_pred.t() @ W_pred).to(torch.float64)
        ev = torch.linalg.eigvalsh(gram)
        gram_conds.append(float(ev[-1] / max(ev[0].item(), 1e-12)))

    diag = {
        "n_built": n_built, "n_skip_layer_not_in_grid": n_skip_layer,
        "n_skip_no_cov": n_skip_nocov, "n_skip_null_dir": n_skip_null,
        "head_dim": head_dim, "k_know": k_know, "k_pred": k_pred,
        "wpred_source": wpred_source,
        "cos_wknow_theta_mean": float(np.mean(cos_wknow_theta)) if cos_wknow_theta else None,
        "cos_wpred_wknow_mean": float(np.mean(cos_wpred_wknow)) if cos_wpred_wknow else None,
        "gram_cond_max": float(max(gram_conds)) if gram_conds else None,
    }
    return head_bases, diag


# ---- Scoring loop ----------------------------------------------------------
def score_arm(model, tokenizer, items, task, apply_ct, label):
    """doc_id -> {'cond':[lp per opt], 'gold':int, 'n_opt':int} for one arm."""
    cfg = PREMISE[task]
    prefix = cfg["prefix"]
    out = {}
    t0 = time.time()
    for i, (did, q, opts, gold) in enumerate(items):
        ctx = build_context(tokenizer, prefix + cfg["cond"].format(q=q), apply_ct)
        cond = [cont_logprob(model, tokenizer, ctx, " " + o) for o in opts]
        out[did] = {"cond": cond, "gold": gold, "n_opt": len(opts)}
        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            el = time.time() - t0
            print(f"  [{label}] {i+1}/{len(items)}  {el:.0f}s  {(i+1)/max(el,1e-9):.1f} it/s",
                  flush=True)
    return out


def acc_from(arm, items, norm: bool):
    """acc (norm=False) or acc_norm (norm=True, char-length-normalized argmax)."""
    hit = []
    for did, q, opts, gold in items:
        lp = np.array(arm[did]["cond"], dtype=np.float64)
        if norm:
            clen = np.array([max(1, len(o)) for o in opts], dtype=np.float64)
            lp = lp / clen
        hit.append(int(int(np.argmax(lp)) == gold))
    return float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--offsets", required=True,
                    help="V7-mc offsets .pt (defines the (layer,head) set KAPPA modulates "
                         "+ supplies the OFF-AXIS arm for the head-to-head)")
    ap.add_argument("--func", default="runs/functional_directions.pt",
                    help="E2 functional_directions.pt (W_know/W_pred source; mu_inf_*, cov_inf)")
    ap.add_argument("--task", required=True, choices=list(PREMISE))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Intermediate layers to apply KAPPA at (KAPPA(k); k=len). "
                         "Only adapter heads on these layers are modulated. DEFAULT "
                         "(None) = the FULL adapter layer set (all layers in the offsets "
                         "blob), so KAPPA covers every off-axis cell and the GATE-2 "
                         "head-set parity holds without restriction. Pass an explicit "
                         "subset only for a deliberate ablation.")
    ap.add_argument("--wpred-source", default="pc1", choices=["pc1", "vinf", "wknow"],
                    help="Prediction-subspace reconstruction (RECONSTRUCTED; default pc1)")
    ap.add_argument("--k-know", type=int, default=1, help="knowledge subspace rank (1 backed)")
    ap.add_argument("--k-pred", type=int, default=1, help="prediction subspace rank")
    ap.add_argument("--kappa-alpha", type=float, default=1.0, help="Eq.3 knowledge amplification")
    ap.add_argument("--kappa-beta", type=float, default=0.0,
                    help="Eq.4 additive SIGN-sharpen term (default 0 = pure Eq.3; UNIFIED 2026-06-24 "
                         "with the faithful script — beta is NO LONGER a multiplier on pred coords)")
    ap.add_argument("--ridge-frac", type=float, default=1e-2,
                    help="LDA/PC ridge as frac of mean cov diag (matches make_wknow_offset.py)")
    ap.add_argument("--skip-offaxis-arm", action="store_true",
                    help="Skip scoring the off-axis V7-mc arm (KAPPA + base only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not (args.base or "").strip():
        raise SystemExit("FATAL: --base empty (fresh pod shell loses $BASE). Re-set it.")

    print("=" * 74, flush=True)
    print(f"KAPPA on-axis FOIL (GATE-2) | task={args.task} limit={args.limit} "
          f"chat_template={args.apply_chat_template}", flush=True)
    _layers_disp = "FULL-adapter-set (resolved after offsets load)" if args.layers is None \
        else f"k={len(args.layers)} layers={args.layers}"
    print(f"  KAPPA({_layers_disp})  wpred={args.wpred_source} "
          f"k_know={args.k_know} k_pred={args.k_pred} alpha={args.kappa_alpha} beta={args.kappa_beta}",
          flush=True)
    print("  REIMPLEMENTATION — Eq.3 FAITHFUL; subspace SPACE (per-head o_proj-input) + "
          "W_know(LDA) + W_pred(recon) DEVIATE. See docstring.", flush=True)
    print("=" * 74, flush=True)

    items = load_items(args.task, args.limit)
    print(f"  {len(items)} items from {CORPUS[args.task]}", flush=True)

    torch_dtype = getattr(torch, args.dtype)
    print(f"  loading base {args.base} ...", flush=True)
    t0 = time.time()
    model = _load_model_multimodal(args.base, torch_dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    # offsets blob defines the (layer,head) set + the off-axis arm
    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    lhp, alpha, theta = blob["layer_head_pairs"], blob["alpha"], blob["theta"]
    layers = _resolve_layers(model)
    attn_shape = _get_attention_shape(model, layers)
    lhp, alpha, theta = _filter_compatible_layers(model, tokenizer, attn_shape, lhp, alpha, theta)
    pairs = [tuple(int(x) for x in p) for p in lhp]
    print(f"  adapter heads (filtered): {len(pairs)}; d_attn={attn_shape.num_q_heads}*"
          f"{attn_shape.head_dim}", flush=True)

    # Resolve --layers default to the FULL adapter layer set (reviewer fix 1):
    # default None -> KAPPA covers every off-axis cell, so the head-to-head is
    # head-set-parity by construction (no 48-head vs ~10-head confound).
    if args.layers is None:
        args.layers = sorted({li for (li, _hi) in pairs})
        print(f"  --layers defaulted to FULL adapter layer set ({len(args.layers)} layers): "
              f"{args.layers}", flush=True)

    # build KAPPA per-head subspaces
    fd = torch.load(args.func, map_location="cpu", weights_only=False)
    only_layers = set(args.layers)
    head_bases, diag = build_head_bases(
        pairs, fd, args.k_know, args.k_pred, args.wpred_source, args.ridge_frac, only_layers)
    print("\n[kappa-subspaces] " + json.dumps(diag), flush=True)
    if diag["n_built"] == 0:
        raise SystemExit(
            "FATAL: 0 KAPPA heads built. Check --layers overlaps adapter layers in the "
            "E2 grid (captured_indices) and that cov_inf has those (layer,head).")
    if diag["cos_wknow_theta_mean"] is not None and abs(diag["cos_wknow_theta_mean"]) > 0.30:
        print(f"  NOTE: |cos(W_know,theta_mc)|={diag['cos_wknow_theta_mean']:.3f} > 0.30 "
              f"(expected ~0 off-axis). Inspect W_know build if surprising.", flush=True)

    # --- BASE arm (no intervention; null LoFiT hooks for codepath parity) ---
    print("\n[arm base] zeroed offsets (identity through the hook path)", flush=True)
    h = install_lofit_hooks(layers, torch.zeros_like(alpha), torch.zeros_like(theta),
                            lhp, attn_shape)
    try:
        base = score_arm(model, tokenizer, items, args.task, args.apply_chat_template, "base")
    finally:
        remove_handles(h)

    # --- KAPPA arm (closed-form Eq.3 projection hooks) ---
    print("\n[arm kappa] closed-form Eq.3 projection installed", flush=True)
    h = install_kappa_hooks(layers, attn_shape, head_bases, args.kappa_alpha, args.kappa_beta)
    try:
        kappa = score_arm(model, tokenizer, items, args.task, args.apply_chat_template, "kappa")
    finally:
        remove_handles(h)

    # --- OFF-AXIS arm (trained V7-mc offset, FULL adapter) for context ---
    offaxis = None
    if not args.skip_offaxis_arm:
        print("\n[arm offaxis] trained V7-mc offset installed (FULL adapter, off-axis "
              "reference)", flush=True)
        h = install_lofit_hooks(layers, alpha, theta, lhp, attn_shape)
        try:
            offaxis = score_arm(model, tokenizer, items, args.task,
                                args.apply_chat_template, "offaxis")
        finally:
            remove_handles(h)

    # --- OFF-AXIS-RESTRICTED arm — head-set parity vs KAPPA (reviewer fix 1) ---
    # Modulate EXACTLY the (layer,head) cells KAPPA built (head_bases keys), so the
    # GATE-2 headline Δ is a same-cells off-axis-vs-on-axis comparison, not a
    # 48-head-vs-~10-head artifact. With --layers=None (full adapter set) the
    # KAPPA-built set == the full adapter set, so this arm == the full off-axis arm.
    offaxis_restricted = None
    alpha_r, theta_r, n_off_restricted, kappa_cells = restrict_offaxis_to_kappa_cells(
        lhp, alpha, theta, head_bases)
    n_kappa_heads = diag["n_built"]
    n_offaxis_full_heads = len(pairs)
    if not args.skip_offaxis_arm:
        print(f"\n[arm offaxis_restricted] off-axis offset zeroed except the "
              f"{n_off_restricted} KAPPA-built cells (head-set parity vs KAPPA's "
              f"{n_kappa_heads} heads)", flush=True)
        h = install_lofit_hooks(layers, alpha_r, theta_r, lhp, attn_shape)
        try:
            offaxis_restricted = score_arm(model, tokenizer, items, args.task,
                                           args.apply_chat_template, "offaxis_restricted")
        finally:
            remove_handles(h)

    # --- metrics + sanity gates ---
    def metrics(arm):
        return None if arm is None else {"acc": acc_from(arm, items, False),
                                         "acc_norm": acc_from(arm, items, True)}
    m_base, m_kappa = metrics(base), metrics(kappa)
    m_off, m_off_r = metrics(offaxis), metrics(offaxis_restricted)
    print("\n" + "=" * 74, flush=True)
    print(f"  head counts: KAPPA={n_kappa_heads}  offaxis_restricted={n_off_restricted}  "
          f"offaxis_FULL={n_offaxis_full_heads}", flush=True)
    if n_off_restricted != n_kappa_heads:
        print(f"  !!! WARNING: offaxis_restricted heads ({n_off_restricted}) != KAPPA heads "
              f"({n_kappa_heads}). Headline Δ would NOT be head-set-parity. Inspect.", flush=True)
    if n_offaxis_full_heads != n_kappa_heads:
        print(f"  CAVEAT: the FULL off-axis arm modulates {n_offaxis_full_heads} heads vs KAPPA's "
              f"{n_kappa_heads} — do NOT use Δ(offaxis_FULL - KAPPA) as the headline (head-count "
              f"confound). Use Δ(offaxis_restricted - KAPPA).", flush=True)
    print(f"  base              : {m_base}", flush=True)
    print(f"  KAPPA             : {m_kappa}   lift acc_norm={m_kappa['acc_norm']-m_base['acc_norm']:+.4f}",
          flush=True)
    if m_off_r is not None:
        print(f"  offaxis_restricted: {m_off_r}   lift acc_norm={m_off_r['acc_norm']-m_base['acc_norm']:+.4f}",
              flush=True)
        print(f"  Δ(offaxis_restricted - KAPPA) acc_norm = "
              f"{m_off_r['acc_norm']-m_kappa['acc_norm']:+.4f}  (GATE-2 HEADLINE — same {n_kappa_heads} cells)",
              flush=True)
    if m_off is not None:
        print(f"  offaxis_FULL      : {m_off}   lift acc_norm={m_off['acc_norm']-m_base['acc_norm']:+.4f} "
              f"  (CONTEXT only — {n_offaxis_full_heads} heads, NOT head-set-parity vs KAPPA)",
              flush=True)
        print(f"  Δ(offaxis_FULL - KAPPA) acc_norm = "
              f"{m_off['acc_norm']-m_kappa['acc_norm']:+.4f}  (NOT headline — head-count confound)",
              flush=True)
    # --- Sanity gate (1): scaffold validity via off-axis conditional-lift (fix 2) ---
    # A broken scaffold (wrong chat-template / continuation tokenization) would make
    # every number silently wrong. We anchor on the FULL off-axis arm reproducing the
    # canonical ~+0.35 ARC / ~+0.33 TQA lift under THIS protocol (sibling
    # score_mc_premise_pod.py uses the same idea). This simultaneously validates the
    # scaffold AND the off-axis install. (If --skip-offaxis-arm, this gate is skipped
    # and we fall back to noting that the base level cannot be cross-checked.)
    canonical_lift = CANONICAL_OFFAXIS_COND_LIFT.get(args.task)
    gate1 = {"task": args.task, "canonical_offaxis_lift": canonical_lift,
             "eps": SANITY_LIFT_EPS, "observed_offaxis_full_lift": None,
             "base_acc_norm": m_base["acc_norm"], "status": None}
    if m_off is not None and canonical_lift is not None:
        off_lift = m_off["acc_norm"] - m_base["acc_norm"]
        gate1["observed_offaxis_full_lift"] = off_lift
        ok = abs(off_lift - canonical_lift) <= SANITY_LIFT_EPS
        gate1["status"] = "OK" if ok else "WARNING"
        print(f"  SANITY GATE (1) scaffold: off-axis FULL lift acc_norm={off_lift:+.4f} "
              f"(expect ~{canonical_lift:+.2f}, eps {SANITY_LIFT_EPS})", flush=True)
        if not ok:
            print(f"  !!! WARNING: off-axis lift {off_lift:+.4f} far from canonical "
                  f"{canonical_lift:+.2f} (|Δ| > {SANITY_LIFT_EPS}). Scaffold likely mismatched "
                  f"(chat-template / continuation) OR offsets/func mismatch — base acc_norm "
                  f"{m_base['acc_norm']:.4f} and ALL downstream numbers are suspect. Inspect "
                  f"before trusting the head-to-head.", flush=True)
        else:
            print(f"  scaffold OK (off-axis lift in range) -> base level {m_base['acc_norm']:.4f} "
                  f"and the head-to-head are trustworthy", flush=True)
    else:
        gate1["status"] = "SKIPPED"
        print(f"  SANITY GATE (1) scaffold: SKIPPED (no off-axis arm or no canonical "
              f"constant for task={args.task}); base acc_norm {m_base['acc_norm']:.4f} "
              f"NOT cross-checked — scaffold unverified.", flush=True)

    # Sanity gate (2): KAPPA must be on-axis non-destructive (paper: never damages ARC).
    kappa_drop = m_kappa["acc_norm"] - m_base["acc_norm"]
    if kappa_drop < -0.05:
        print(f"  !!! WARNING: KAPPA acc_norm DROP {kappa_drop:+.4f} (<-0.05). The paper reports "
              f"KAPPA never damages ARC — a large drop suggests a degenerate subspace or an Eq.3 "
              f"bug. Inspect probe diagnostics before trusting.", flush=True)
    else:
        print(f"  KAPPA non-destructive check OK (Δacc_norm {kappa_drop:+.4f} >= -0.05)", flush=True)
    print("=" * 74, flush=True)

    payload = {
        "task": args.task,
        "method": "KAPPA_reimpl_eq3_perhead_oproj_input",
        "paper": "Park,Pyun,Jo arXiv:2509.23782 ICML2026",
        "faithful_vs_reconstructed": {
            "eq3_algebra": "FAITHFUL",
            "applied_intermediate_layers": "FAITHFUL",
            "closed_form_no_training": "FAITHFUL",
            "scoring_protocol_parity": "FAITHFUL",
            "subspace_space_per_head_oproj_input": "RECONSTRUCTED (paper=residual d_model)",
            "W_know_source_fisher_lda_groundtruth": "RECONSTRUCTED (paper=k-class option-truth probe)",
            "W_pred_source_" + args.wpred_source: "RECONSTRUCTED (paper=greedy-option probe)",
            "alpha_beta_sharpen": f"alpha={args.kappa_alpha},beta={args.kappa_beta} (defaults 1.0; paper exact values not transcribed)",
        },
        "config": {
            "layers": args.layers, "wpred_source": args.wpred_source,
            "k_know": args.k_know, "k_pred": args.k_pred,
            "kappa_alpha": args.kappa_alpha, "kappa_beta": args.kappa_beta,
            "ridge_frac": args.ridge_frac, "apply_chat_template": args.apply_chat_template,
            "offsets": args.offsets, "func": args.func, "limit": args.limit,
        },
        "subspace_diagnostics": diag,
        # Head-set parity (reviewer fix 1): the GATE-2 headline Δ MUST be
        # offaxis_restricted (same cells as KAPPA) vs KAPPA — NOT offaxis_FULL
        # (different head count). These counts let the local assembly assert parity.
        "head_set_parity": {
            "n_kappa_heads": n_kappa_heads,
            "n_offaxis_restricted_heads": n_off_restricted,
            "n_offaxis_full_heads": n_offaxis_full_heads,
            "kappa_cells": sorted([list(c) for c in kappa_cells]),
            "headline_delta_arm": "offaxis_restricted",
            "headline_delta_acc_norm": (None if m_off_r is None
                                        else m_off_r["acc_norm"] - m_kappa["acc_norm"]),
            "offaxis_full_minus_kappa_acc_norm_NOT_HEADLINE": (
                None if m_off is None else m_off["acc_norm"] - m_kappa["acc_norm"]),
        },
        # Scaffold sanity gate (1) (reviewer fix 2): off-axis FULL conditional lift
        # must reproduce the canonical ~+0.35 ARC / +0.33 TQA, else the scaffold is
        # broken and ALL numbers below are suspect.
        "sanity_gate_1_scaffold": gate1,
        "metrics": {"base": m_base, "kappa": m_kappa,
                    "offaxis_restricted": m_off_r, "offaxis_full": m_off,
                    # back-compat alias: 'offaxis' == the FULL arm (context, NOT headline)
                    "offaxis": m_off},
        "gold": {str(did): gold for did, q, opts, gold in items},
        "charlen": {str(did): [max(1, len(o)) for o in opts] for did, q, opts, gold in items},
        # Per-item per-option conditional log-probs for the local head-to-head +
        # PMI-residual / per-item collateral analysis. doc_ids align with the
        # COMMITTED CORPUS (CORPUS[task] = runs/correctness_probe_postgen/corpus_*.jsonl).
        # The head-to-head MUST use THESE internally re-scored arms (base/kappa/
        # offaxis_*), NOT the raw d_*.json acc numbers — d_*.json is an lm-eval
        # thinking-mode run (different scaffold + no-leading-space continuation), so
        # only the doc_id INDEX is shared with it, never the logprob values. PMI-
        # residual item selection must be recomputed under THIS script's protocol.
        "conditional": {
            "base": {str(did): base[did]["cond"] for did, *_ in items},
            "kappa": {str(did): kappa[did]["cond"] for did, *_ in items},
            **({"offaxis_restricted": {str(did): offaxis_restricted[did]["cond"]
                                       for did, *_ in items}}
               if offaxis_restricted is not None else {}),
            **({"offaxis_full": {str(did): offaxis[did]["cond"] for did, *_ in items},
                # back-compat alias: 'offaxis' == the FULL arm
                "offaxis": {str(did): offaxis[did]["cond"] for did, *_ in items}}
               if offaxis is not None else {}),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {args.out}", flush=True)
    print("  next: assemble GATE-2 head-to-head (off-axis vs KAPPA vs DC-PMI vs DoLa) locally; "
          "report acc_norm recovery + MMLU/EOS collateral per arm; ARC/TQA never pooled.",
          flush=True)


if __name__ == "__main__":
    main()
