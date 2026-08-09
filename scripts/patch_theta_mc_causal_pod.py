"""B12 CAUSAL confirmation — is the +35 offset (theta_mc) a temperature mechanism or a
ranking change?  Real-forward upgrade of the DLA approximation.  GPU / pod.

WHY (2026-06-30): the null-space test
(analyze_theta_mc_null_vs_row.py) REFUTED the Stolfo confidence-neuron / temperature
mechanism, but via DIRECT-LOGIT-ATTRIBUTION on the raw write vector — a LINEARIZED
approximation (fixed RMSNorm gain, direct path only, no inter-layer reprocessing, 1 run).
This script confirms/reopens that CAUSALLY: it runs the REAL model on REAL MC prompts and
measures the ACTUAL output-logit delta base->adapter.

⚠ POSITION MATTERS (lesson from the first run, 2026-07-01):
  --positions option    (DEFAULT) — logit delta at the positions that PREDICT the gold
    option-continuation tokens. THIS is where the +35 acc_norm lift lives -> the on-target
    B12-scoring test (temperature vs ranking OF THE SCORING).
  --positions next_token — logit delta at the single position after the prompt ('Answer:').
    That is the GENERATION face (what the model emits next), NOT the scoring. The first run
    used this and found strong flattening (gamma -0.59, d_entropy +0.99, resid_norm 0.64) —
    a real signal, but it is the §5.6 structural-suppression / EOS-collapse face, NOT the
    scoring mechanism. Kept as an option to reproduce that (separate) result.

DECOMPOSITION (orthogonal; 1_vocab ⊥ z_centered by construction):
    delta = mean(delta)*1        (uniform-additive, softmax-invariant)          frac_uniform
          + gamma*z_centered     (multiplicative temperature; Stolfo signature)  frac_temp
          + ranking_residual     (⊥ both -> genuine re-ranking of tokens)         frac_ranking
gamma is the effective (c-1) temperature coefficient; cos(delta, z_centered) = signed frac_temp.
When multiple positions are scored (option mode) the per-position fracs are averaged per item.

PREDICTION (if B12-scoring holds): at the OPTION positions, frac_temp LOW, frac_ranking HIGH,
|gamma| small -> the offset moves token DIFFERENCES in the scoring, not a temperature.

Emits per-item scalars for the LOCAL bootstrap in analyze_theta_mc_patch.py.

NOTE: Stage A (check_canonical_repro) must have PASSED in the same pod (guards BASE/offsets).

Usage (pod, GPU; cd /workspace/MSAP per block):
  BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' \
        2>/dev/null | grep -v '/.locks/' | head -1)")
  python scripts/patch_theta_mc_causal_pod.py --base "$BASE" \
      --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --task arc_challenge --limit 400 --apply-chat-template --positions option \
      --out runs/mechanism_v8/theta_mc_patch_arc_option.json
  # LOCAL: python scripts/analyze_theta_mc_patch.py --in runs/mechanism_v8/theta_mc_patch_arc_option.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.score_mc_premise_pod import load_items, build_context, PREMISE, CORPUS
from scripts.run_lm_eval_v7 import _load_model_multimodal, _filter_compatible_layers
from scripts.eval_with_router import (
    install_lofit_hooks, remove_handles, _resolve_layers, _get_attention_shape,
)
from transformers import AutoTokenizer

_CAP = {"resid": None}


def _find_final_norm(model):
    """Locate the final RMSNorm (input = residual pre-unembedding). Robust to multimodal nesting."""
    for path in ("model.norm", "model.model.norm", "language_model.model.norm",
                 "model.language_model.model.norm", "model.language_model.norm"):
        m = model
        ok = True
        for p in path.split("."):
            if hasattr(m, p):
                m = getattr(m, p)
            else:
                ok = False
                break
        if ok and m is not model:
            return m, path
    return None, None


def _capture_hook(_mod, inputs):
    _CAP["resid"] = inputs[0].detach()   # (B, T, d_model)


@torch.no_grad()
def logits_resid_at(model, ids, positions):
    """Run forward on `ids` (1,T); return (logits[positions] float32 (P,vocab),
    mean residual-norm over `positions` or None)."""
    _CAP["resid"] = None
    out = model(ids, use_cache=False)
    z = out.logits[0, positions].float()          # (P, vocab)
    rn = None
    if _CAP["resid"] is not None:
        rn = float(_CAP["resid"][0, positions].float().norm(dim=-1).mean().item())
    return z, rn


def decompose_one(z_base: torch.Tensor, z_adapter: torch.Tensor, fc: torch.Tensor | None = None) -> dict:
    """Orthogonal split of one logit delta into uniform(||1) / temp(∝z_centered) / ranking.

    If fc (mean-centered unigram log-freq, ⊥ 1_vocab by construction) is given, also
    reports the FREQUENCY component of the causal delta — the C-4 cross-check of the
    direct-logit cos(agg, d_freq)=0.0128 (audit R-3: temperature got a causal check
    that flipped its sign; frequency was direct-logit-only until this).
      cos_freq       — cos(delta, fc): same construction as cos_temp w.r.t. z_centered.
      frac_freq      — |proj_fc(delta)| / |delta|.
      *_perp variants— fc orthogonalized against z_centered too (additive-clean split)."""
    delta = z_adapter - z_base
    V = delta.shape[0]
    total = float(delta.norm())
    if total < 1e-9:
        out = {"frac_uniform": 0.0, "frac_temp": 0.0, "frac_ranking": 0.0,
               "gamma": 0.0, "cos_temp": 0.0, "d_entropy": 0.0, "argmax_changed": 0}
        if fc is not None:
            out.update({"cos_freq": 0.0, "frac_freq": 0.0,
                        "cos_freq_perp": 0.0, "frac_freq_perp": 0.0})
        return out
    mean = float(delta.mean())
    uniform_norm = abs(mean) * math.sqrt(V)
    zc = z_base - z_base.mean()
    zc_sq = float((zc * zc).sum())
    gamma = float((delta * zc).sum() / zc_sq) if zc_sq > 1e-9 else 0.0
    temp_vec = gamma * zc
    ranking_vec = delta - mean - temp_vec
    hb = float(-(torch.softmax(z_base, -1) * torch.log_softmax(z_base, -1)).sum())
    ha = float(-(torch.softmax(z_adapter, -1) * torch.log_softmax(z_adapter, -1)).sum())
    out = {"frac_uniform": uniform_norm / total, "frac_temp": float(temp_vec.norm()) / total,
           "frac_ranking": float(ranking_vec.norm()) / total, "gamma": gamma,
           "cos_temp": float((delta * zc).sum() / (delta.norm() * (zc.norm() + 1e-9))),
           "d_entropy": ha - hb, "argmax_changed": int(int(z_base.argmax()) != int(z_adapter.argmax()))}
    if fc is not None:
        fcn = float(fc.norm())
        dot_f = float((delta * fc).sum())
        out["cos_freq"] = dot_f / (total * fcn + 1e-9)
        out["frac_freq"] = abs(dot_f) / (fcn * total + 1e-9)  # == |cos_freq| (rank-1 proj, mirrors frac_temp==|cos_temp|)
        if zc_sq > 1e-9:
            fperp = fc - (float((fc * zc).sum()) / zc_sq) * zc
        else:
            fperp = fc
        fpn = float(fperp.norm())
        dot_fp = float((delta * fperp).sum())
        out["cos_freq_perp"] = dot_fp / (total * fpn + 1e-9)
        out["frac_freq_perp"] = abs(dot_fp) / (fpn * total + 1e-9)
    return out


def decompose_positions(zb, za, fc: torch.Tensor | None = None):
    """Average the per-position decomposition (option mode scores several continuation tokens)."""
    P = zb.shape[0]
    ds = [decompose_one(zb[p], za[p], fc) for p in range(P)]
    keys = list(ds[0].keys())
    return {k: float(sum(d[k] for d in ds) / max(P, 1)) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--task", required=True, choices=list(PREMISE))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--positions", choices=["option", "next_token"], default="option",
                    help="option=gold-continuation scoring positions (B12-scoring, DEFAULT); "
                         "next_token=post-prompt position (generation face / §5.6)")
    ap.add_argument("--freq-vector", type=Path, default=None,
                    help="d_freq_vector.pt from derive_freq_control.py --save-vector (must contain "
                         "'f' = unigram log-freq). Adds the frequency component to the causal "
                         "decomposition — the C-4 cross-check of the direct-logit cos 0.0128 (R-3).")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not (args.base or "").strip():
        raise SystemExit("FATAL: --base empty (fresh pod shell loses $BASE). Re-set it.")

    print("=" * 72, flush=True)
    print(f"B12 causal patch | task={args.task} limit={args.limit} positions={args.positions} "
          f"chat_template={args.apply_chat_template}", flush=True)
    print("=" * 72, flush=True)

    items = load_items(args.task, args.limit)
    cfg = PREMISE[args.task]
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

    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    lhp, alpha, theta = blob["layer_head_pairs"], blob["alpha"], blob["theta"]
    layers = _resolve_layers(model)
    attn_shape = _get_attention_shape(model, layers)
    lhp, alpha, theta = _filter_compatible_layers(model, tokenizer, attn_shape, lhp, alpha, theta)

    fn, fn_path = _find_final_norm(model)
    cap_handle = fn.register_forward_pre_hook(_capture_hook) if fn is not None else None
    print(f"  final-norm capture: {'ON @ '+fn_path if cap_handle else 'OFF (resid-norm skipped)'}", flush=True)

    fc_freq = None
    freq_meta = None
    if args.freq_vector is not None:
        fblob = torch.load(args.freq_vector, map_location="cpu", weights_only=False)
        if "f" not in fblob:
            raise SystemExit("FATAL: --freq-vector blob has no 'f' — re-run "
                             "derive_freq_control.py --save-vector with the 2026-07-23 version.")
        f_log = fblob["f"].to(torch.float32)
        fc_freq = (f_log - f_log.mean()).to(model.device)
        freq_meta = {"vector": str(args.freq_vector),
                     "ident_corr": float(fblob.get("ident_corr", float("nan"))),
                     "wikitext_tokens_seen": int(fblob.get("wikitext_tokens_seen", -1)),
                     "ridge": float(fblob.get("ridge", float("nan"))),
                     "cos_agg_dfreq_direct_logit": 0.0128}
        print(f"  freq signature loaded: vocab={f_log.numel()} ident_corr={freq_meta['ident_corr']:.4f} "
              f"tokens={freq_meta['wikitext_tokens_seen']/1e6:.1f}M ridge={freq_meta['ridge']}", flush=True)

    # Pre-build (ids, positions) per item for the chosen mode.
    prepared = []
    for did, q, opts, gold in items:
        ctx = build_context(tokenizer, cfg["prefix"] + cfg["cond"].format(q=q), args.apply_chat_template)
        if args.positions == "option":
            ctx_ids = tokenizer(ctx, add_special_tokens=False).input_ids
            full = tokenizer(ctx + " " + opts[gold], add_special_tokens=False).input_ids
            n_ctx = len(ctx_ids)
            pos = list(range(n_ctx - 1, len(full) - 1)) or [len(full) - 1]   # predict continuation tokens
            ids = torch.tensor([full], device=model.device)
        else:  # next_token
            enc = tokenizer(ctx, add_special_tokens=False).input_ids
            ids = torch.tensor([enc], device=model.device)
            pos = [len(enc) - 1]
        prepared.append((did, ids, pos))

    per_item = []
    t0 = time.time()
    for i, (did, ids, pos) in enumerate(prepared):
        h = install_lofit_hooks(layers, torch.zeros_like(alpha), torch.zeros_like(theta), lhp, attn_shape)
        try:
            zb, rnb = logits_resid_at(model, ids, pos)
        finally:
            remove_handles(h)
        h = install_lofit_hooks(layers, alpha, theta, lhp, attn_shape)
        try:
            za, rna = logits_resid_at(model, ids, pos)
        finally:
            remove_handles(h)
        d = decompose_positions(zb, za, fc_freq)
        d["doc_id"] = int(did)
        d["n_pos"] = int(zb.shape[0])
        d["resid_norm_base"] = rnb
        d["resid_norm_adapter"] = rna
        d["resid_norm_ratio"] = (rna / rnb) if (rnb and rnb > 0) else None
        per_item.append(d)
        if (i + 1) % 25 == 0 or (i + 1) == len(prepared):
            el = time.time() - t0
            print(f"  [{i+1}/{len(prepared)}] {el:.0f}s  {(i+1)/max(el,1e-9):.1f} it/s", flush=True)

    if cap_handle is not None:
        cap_handle.remove()

    def mean(key):
        vals = [p[key] for p in per_item if p.get(key) is not None]
        return sum(vals) / max(len(vals), 1)
    fu, ft, fr = mean("frac_uniform"), mean("frac_temp"), mean("frac_ranking")
    print("\n" + "=" * 72, flush=True)
    print(f"  SUMMARY (positions={args.positions}, {len(per_item)} items, "
          f"mean n_pos={mean('n_pos'):.1f}):", flush=True)
    print(f"    frac_uniform={fu:.4f}  frac_temp(∝z)={ft:.4f}  frac_ranking={fr:.4f}", flush=True)
    print(f"    mean gamma={mean('gamma'):+.5f}  mean d_entropy={mean('d_entropy'):+.4f}  "
          f"argmax_changed={mean('argmax_changed'):.3f}", flush=True)
    if any(p.get("resid_norm_ratio") for p in per_item):
        print(f"    resid_norm_ratio(adapter/base)={mean('resid_norm_ratio'):.5f}", flush=True)
    if fc_freq is not None:
        print(f"    frac_freq(∝f)={mean('frac_freq'):.4f}  cos_freq={mean('cos_freq'):+.4f}  "
              f"frac_freq_perp={mean('frac_freq_perp'):.4f}  cos_freq_perp={mean('cos_freq_perp'):+.4f}", flush=True)
        print(f"    direct-logit freq reference: cos(agg,d_freq)=+0.0128 (freq_control_r1e3_full)", flush=True)
    print(f"    DLA reference: uniform_frac 0.301 / cos(agg,d*) 0.023 -> RANKING-DOMINANT", flush=True)
    print("=" * 72, flush=True)

    payload = {
        "task": args.task, "n": len(per_item), "positions": args.positions,
        "apply_chat_template": args.apply_chat_template, "final_norm_captured": cap_handle is not None,
        "summary": {"frac_uniform": fu, "frac_temp": ft, "frac_ranking": fr,
                    "gamma": mean("gamma"), "d_entropy": mean("d_entropy"),
                    "argmax_changed": mean("argmax_changed")},
        "dla_reference": {"uniform_frac": 0.301, "cos_agg_dstar": 0.023, "verdict": "RANKING-DOMINANT"},
        "per_item": per_item,
        "note": "Orthogonal split delta = uniform(||1) + gamma*z_centered(temp) + ranking. "
                "positions=option -> B12-scoring (gold-continuation positions); "
                "positions=next_token -> generation/§5.6 face. Bootstrap + verdict LOCAL.",
    }
    if fc_freq is not None:
        payload["summary"].update({"frac_freq": mean("frac_freq"), "cos_freq": mean("cos_freq"),
                                   "frac_freq_perp": mean("frac_freq_perp"),
                                   "cos_freq_perp": mean("cos_freq_perp")})
        payload["freq_reference"] = freq_meta
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {args.out}", flush=True)
    print(f"  next LOCAL: python scripts/analyze_theta_mc_patch.py --in {args.out}", flush=True)


if __name__ == "__main__":
    main()
