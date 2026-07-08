"""nnsight CAUSAL patch of theta_mc on REAL gsm8k GENERATION positions — the gen face (P6).

WHY (audit-remediation P6, decided 2026-07-07 post-B25):
  The plan (2026-07-02) specified P6 as a per-token LINEARIZED DLA (logit_lens_theta_mc).
  B25 (2026-07-07) then showed causally that the DLA/logit-lens signature is MISLEADING on
  the scoring face: per-position cos(delta_causal, delta_DLA) = 0.09/0.19 < null — the offset's
  real effect is dominated by downstream non-linearity the DLA cannot see; the robust signature
  was STRUCTURE-SUPPRESSION read out from the causal patch (nnsight), not the DLA. B25 covered
  the SCORING face only. This script carries the same causal instrument to the GENERATION face,
  which B25 explicitly deferred ("nnsight's generation instrumentation is the killer feature the
  hooks lack").

THE QUESTION P6 DECIDES (ledger B17, the "commitment machine" conjecture — today OUTSIDE the draft):
  Over a real gsm8k generation, does the offset SUPPRESS format/structure tokens (punct, markdown,
  capitalization, whitespace) COHERENTLY and does that suppression GROW along the generation
  (early tokens ~ untouched, late tokens ~ increasingly de-structured)?
    coherent AND growing -> the commitment-machine frame earns its number -> B17 enters (via ledger)
    genuine flattening with NO structure signature -> B17 dies (conjecture refuted)
    mixed -> stays out of the draft (status quo)
  Extra discriminator: loops (non-terminating) vs clean-wrong (terminate with wrong ####). If the
  loops show STRONGER / more monotone structure-suppression than clean-wrong, that is consistent
  with "the offset keeps de-structuring so the trace never commits to a close" (loops) vs "the
  trace commits but to a wrong number" (clean-wrong).

METHOD (causal, per generated position; mirrors patch_theta_mc_nnsight_pod M1 but on gen positions):
  1. Rebuild the 26 loops + 24 clean-wrong items from the R6probe_v7mc forensic (exact_match==0;
     '####' absent => loop, present => clean-wrong).
  2. Generate WITH the adapter (install_lofit_hooks at full alpha, greedy) to get the REAL adapter
     token ids (the positions where the -25 pathology actually happens, loops included).
  3. Over [prompt + adapter_gen], forward base (zb) and nnsight output-patched adapter (za); the
     per-position delta = za - zb.  (o_proj is linear => patching r_layer into o_proj.output is
     identical to install_lofit_hooks, cos~1 — the same equivalence B25 SANITY-checked.)
  4. Vocabulary category mask (categorize from logit_lens): STRUCTURE = punct/symbol, whitespace,
     Capitalized_word, single_letter; CONTENT = digit, word, mixed/subword. Per position:
     structure_frac = suppression-mass on STRUCTURE / (STRUCTURE + CONTENT). Bucket positions by
     generation-progress quartile -> trajectory. Also the B25-comparable top-k suppressed
     category-mix (aggregate) so the gen face can be read against the scoring face directly.

SANITY: same nnsight==hooks equivalence (cos>=0.99) as B25 on a few positions; scale=0 delta ~ 0.

Usage (pod, GPU; cd /workspace/MSAP per block; needs `pip install nnsight`):
  BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' \
        2>/dev/null | grep -v '/.locks/' | head -1)")
  python scripts/patch_theta_mc_nnsight_gen_pod.py --base "$BASE" \
      --offsets runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt \
      --r6probe runs/eval_bulletproof/R6probe_v7mc_cap512.json \
      --n-loops 26 --n-cleanwrong 24 --max-new-tokens 512 \
      --out runs/mechanism_v8/theta_mc_nnsight_gen_gsm8k.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

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
from scripts.logit_lens_theta_mc import categorize
from scripts.patch_theta_mc_nnsight_pod import (
    _resolve_layers_path, _walk, _compute_r_layers, _topk, _cos,
)
from scripts.gen_blend_decoding_gsm8k_pod import (
    STOP_IDS, build_fewshot, build_prompt,
)
from transformers import AutoTokenizer

TOPK = 30
STRUCTURE_CATS = {"punct/symbol", "whitespace", "Capitalized_word", "single_letter"}
CONTENT_CATS = {"digit", "word", "mixed/subword"}


# ---------------------------------------------------------------------------- item rebuild
def _rebuild_forensic_items(r6probe_path, n_loops, n_cleanwrong):
    """From the R6probe_v7mc lm-eval dump, pick n_loops non-terminating + n_cleanwrong
    terminating-but-wrong gsm8k items. Returns list of {doc_id, question, kind}."""
    d = json.load(open(r6probe_path, encoding="utf-8"))
    task = next(iter(d["samples"]))
    samples = d["samples"][task]
    loops, cleanwrong, seen = [], [], set()
    for s in samples:
        did = s["doc_id"]
        if did in seen:                      # cap512 dump can repeat docs across filters
            continue
        seen.add(did)
        em = float(s.get("exact_match", 0.0))
        gen = s["resps"][0][0] if s.get("resps") else ""
        q = s["doc"]["question"]
        if em >= 0.5:
            continue                         # correct — not forensic material
        if "####" in gen:
            cleanwrong.append({"doc_id": did, "question": q, "kind": "clean_wrong"})
        else:
            loops.append({"doc_id": did, "question": q, "kind": "loop"})
    picked = loops[:n_loops] + cleanwrong[:n_cleanwrong]
    print(f"[gen] forensic pool: {len(loops)} loops / {len(cleanwrong)} clean-wrong available; "
          f"picked {min(n_loops,len(loops))} + {min(n_cleanwrong,len(cleanwrong))} = {len(picked)}",
          flush=True)
    return picked


# ---------------------------------------------------------------------------- vocab mask
def _category_masks(tokenizer, vocab):
    """(is_structure, is_content) bool tensors over the vocab, via logit_lens categorize."""
    is_struct = torch.zeros(vocab, dtype=torch.bool)
    is_cont = torch.zeros(vocab, dtype=torch.bool)
    toks = tokenizer.convert_ids_to_tokens(list(range(vocab)))
    for i, t in enumerate(toks):
        if t is None:
            continue
        c = categorize(t)
        if c in STRUCTURE_CATS:
            is_struct[i] = True
        elif c in CONTENT_CATS:
            is_cont[i] = True
    print(f"[gen] vocab mask: {int(is_struct.sum())} structure / {int(is_cont.sum())} content "
          f"/ {vocab} total", flush=True)
    return is_struct, is_cont


# ---------------------------------------------------------------------------- generation
@torch.no_grad()
def _generate_adapter(model, tok, prompts, max_new_tokens, batch_size,
                      layers, alpha, theta, lhp, attn_shape):
    """Greedy generation WITH the adapter installed (full alpha). Returns list of gen token-id
    lists (the real adapter trace positions, loops included). Left-padded batches; strip pad."""
    handles = install_lofit_hooks(layers, alpha, theta, lhp, attn_shape)
    gens = []
    try:
        n = len(prompts)
        for bi in range(0, n, batch_size):
            chunk = prompts[bi:bi + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            in_len = enc["input_ids"].shape[1]
            out = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                eos_token_id=STOP_IDS, pad_token_id=tok.pad_token_id,
            )
            new = out[:, in_len:]
            for j in range(len(chunk)):
                row = new[j]
                keep = row[row != tok.pad_token_id]
                gens.append(keep.tolist())
            print(f"    [gen adapter] [{bi + len(chunk)}/{n}]", flush=True)
    finally:
        remove_handles(handles)
    return gens


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--r6probe", default="runs/eval_bulletproof/R6probe_v7mc_cap512.json")
    ap.add_argument("--n-loops", type=int, default=26)
    ap.add_argument("--n-cleanwrong", type=int, default=24)
    ap.add_argument("--k-shot", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--offset-scale", type=float, default=1.0,
                    help="multiply alpha (run_lm_eval_v7 semantics); 1.0 = baked strength")
    ap.add_argument("--sanity-n", type=int, default=3)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--quartiles", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not (args.base or "").strip():
        raise SystemExit("FATAL: --base empty (fresh pod shell loses $BASE). Re-set it.")

    print("=" * 78, flush=True)
    print(f"nnsight causal patch — GEN FACE (P6) | loops={args.n_loops} clean_wrong={args.n_cleanwrong} "
          f"max_new={args.max_new_tokens} scale={args.offset_scale}", flush=True)
    print("=" * 78, flush=True)

    try:
        from nnsight import LanguageModel
    except Exception as e:
        raise SystemExit(f"FATAL: nnsight import failed ({e}). In the pod: pip install nnsight") from e

    forensic = _rebuild_forensic_items(args.r6probe, args.n_loops, args.n_cleanwrong)
    if not forensic:
        raise SystemExit("FATAL: no forensic items rebuilt — check --r6probe path/structure.")

    torch_dtype = getattr(torch, args.dtype)
    print(f"  loading base {args.base} ...", flush=True)
    t0 = time.time()
    hf_model = _load_model_multimodal(args.base, torch_dtype)
    hf_model.eval()
    hf_model.requires_grad_(False)                # inference-only (nnsight would else OOM)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = hf_model.device
    print(f"  loaded in {time.time()-t0:.0f}s  device={device}", flush=True)

    # -- offsets + compatible filter + scale --------------------------------------
    blob = torch.load(args.offsets, map_location="cpu", weights_only=False)
    lhp, alpha, theta = blob["layer_head_pairs"], blob["alpha"], blob["theta"]
    layers = _resolve_layers(hf_model)
    attn_shape = _get_attention_shape(hf_model, layers)
    lhp, alpha, theta = _filter_compatible_layers(hf_model, tok, attn_shape, lhp, alpha, theta)
    alpha = alpha.float() * float(args.offset_scale)
    adapted_layers = sorted({int(L) for L, _ in lhp})
    print(f"  {len(lhp)} adapted heads over {len(adapted_layers)} layers: {adapted_layers}", flush=True)

    r_layers_cpu, _ = _compute_r_layers(hf_model, layers, lhp, alpha, theta, attn_shape)
    r_layers = {L: r.to(device=device, dtype=torch_dtype) for L, r in r_layers_cpu.items()}

    # -- build prompts + generate the REAL adapter trace --------------------------
    fewshot = build_fewshot(tok, args.k_shot)
    prompts = [build_prompt(tok, fewshot, it["question"]) for it in forensic]
    print(f"  generating adapter traces for {len(prompts)} forensic prompts ...", flush=True)
    gen_ids = _generate_adapter(hf_model, tok, prompts, args.max_new_tokens,
                                args.gen_batch_size, layers, alpha, theta, lhp, attn_shape)

    # -- prompt token ids. build_prompt renders the canonical gsm8k few-shot prompt (the exact
    #    string used in the -25 R6 run); it already emitted any special tokens, so tokenize the
    #    rendered string WITHOUT re-adding them (add_special_tokens=False) — same convention as
    #    the gsm8k gen harness (gen_gsm8k_scale_pod). ----
    prompt_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]

    lm = LanguageModel(hf_model, tokenizer=tok)
    layers_path = _resolve_layers_path(hf_model)
    print(f"  nnsight wrapped; decoder-layers path = '{layers_path}'", flush=True)

    def _base_logits(ids, pos):
        with torch.no_grad():
            return hf_model(ids, use_cache=False).logits[0, pos].float().cpu()

    def _patched_logits(ids, pos):
        with torch.no_grad(), lm.trace(ids, scan=False, validate=False):
            for L in adapted_layers:
                o_proj = _walk(lm, layers_path)[L].self_attn.o_proj
                cur = o_proj.output
                o_proj.output[:] = cur + r_layers[L]
            saved = lm.output.logits[0, pos].save()
        return saved.float().cpu()

    vocab = hf_model.get_input_embeddings().weight.shape[0]   # tied unembed (always present)
    is_struct, is_cont = _category_masks(tok, vocab)

    # === SANITY: nnsight==hooks on a few gen positions ============================
    print("\n" + "-" * 78, flush=True)
    print(f"  SANITY equivalence (nnsight vs hooks) on {args.sanity_n} items ...", flush=True)
    cos_eq = []
    for it_i in range(min(args.sanity_n, len(forensic))):
        full = prompt_ids[it_i] + gen_ids[it_i]
        n_ctx = len(prompt_ids[it_i])
        pos = list(range(n_ctx - 1, len(full) - 1)) or [len(full) - 1]
        ids = torch.tensor([full], device=device)
        zb = _base_logits(ids, pos)
        h = install_lofit_hooks(layers, alpha, theta, lhp, attn_shape)
        try:
            with torch.no_grad():
                z_hooks = hf_model(ids, use_cache=False).logits[0, pos].float().cpu()
        finally:
            remove_handles(h)
        z_ns = _patched_logits(ids, pos)
        cos_eq.append(_cos(z_ns - zb, z_hooks - zb))
    cos_eq_mean = sum(cos_eq) / max(len(cos_eq), 1)
    print(f"    mean cos(delta_nnsight, delta_hooks) = {cos_eq_mean:.5f}  {[round(c,4) for c in cos_eq]}",
          flush=True)
    if cos_eq_mean < 0.99:
        print("    ⚠ WARNING: equivalence < 0.99 — do NOT trust the gen deltas; investigate.", flush=True)

    # === per-position causal delta over generated positions ======================
    print("\n" + "-" * 78, flush=True)
    print(f"  measuring causal delta over generated positions, {len(forensic)} items ...", flush=True)
    nq = args.quartiles
    # accumulators: [kind][quartile] -> (sum_structure_frac, count) and suppressed cat counter
    kinds = ["loop", "clean_wrong"]
    q_frac_sum = {k: [0.0] * nq for k in kinds}
    q_frac_cnt = {k: [0] * nq for k in kinds}
    q_supp_struct = {k: [0.0] * nq for k in kinds}   # raw structure suppression mass
    q_supp_cont = {k: [0.0] * nq for k in kinds}
    supp_cat_topk = {k: Counter() for k in kinds}    # B25-comparable top-k suppressed cat-mix
    gen_len_by_kind = {k: [] for k in kinds}
    per_item = []
    t0 = time.time()
    for i, it in enumerate(forensic):
        kind = it["kind"]
        full = prompt_ids[i] + gen_ids[i]
        n_ctx = len(prompt_ids[i])
        gpos = list(range(n_ctx - 1, len(full) - 1))
        if not gpos:
            continue
        ids = torch.tensor([full], device=device)
        zb = _base_logits(ids, gpos)                 # (P, vocab)
        za = _patched_logits(ids, gpos)
        d = za - zb                                  # (P, vocab) fp32 cpu; suppression = -d>0
        supp = (-d).clamp(min=0.0)                   # suppression mass per token
        supp_struct_pos = supp[:, is_struct].sum(1)  # (P,)
        supp_cont_pos = supp[:, is_cont].sum(1)      # (P,)
        denom = (supp_struct_pos + supp_cont_pos).clamp(min=1e-9)
        frac_pos = (supp_struct_pos / denom)         # (P,) structure share of suppression
        P = d.shape[0]
        gen_len_by_kind[kind].append(P)
        # quartile bucket by normalized position along the generation
        for p in range(P):
            qb = min(nq - 1, int(p * nq / P))
            q_frac_sum[kind][qb] += float(frac_pos[p])
            q_frac_cnt[kind][qb] += 1
            q_supp_struct[kind][qb] += float(supp_struct_pos[p])
            q_supp_cont[kind][qb] += float(supp_cont_pos[p])
        # aggregate top-k suppressed category mix over this item's positions (B25-comparable)
        d_sum = d.sum(0)                             # (vocab,) net delta over positions
        _, supp_toks = _topk(d_sum, tok, args.topk)
        for t_str, _v in supp_toks:
            supp_cat_topk[kind][categorize(t_str)] += 1
        per_item.append({"doc_id": it["doc_id"], "kind": kind, "gen_len": P,
                         "struct_frac_mean": round(float(frac_pos.mean()), 4)})
        if (i + 1) % 5 == 0 or (i + 1) == len(forensic):
            print(f"    [{i+1}/{len(forensic)}] {time.time()-t0:.0f}s", flush=True)

    # === assemble trajectory + gate ==============================================
    def _q_series(kind):
        return [round(q_frac_sum[kind][q] / max(q_frac_cnt[kind][q], 1), 4) for q in range(nq)]

    traj = {k: _q_series(k) for k in kinds}
    traj["all"] = [
        round(sum(q_frac_sum[k][q] for k in kinds) / max(sum(q_frac_cnt[k][q] for k in kinds), 1), 4)
        for q in range(nq)
    ]
    # slope Q_last - Q_first (growing structure-suppression along the generation)
    slope = {k: round(v[-1] - v[0], 4) for k, v in traj.items()}
    # coherence: mean structure share over the whole generation (all positions)
    tot_struct = sum(sum(q_supp_struct[k]) for k in kinds)
    tot_cont = sum(sum(q_supp_cont[k]) for k in kinds)
    coherence_frac = round(tot_struct / max(tot_struct + tot_cont, 1e-9), 4)
    supp_cat_all = Counter()
    for k in kinds:
        supp_cat_all.update(supp_cat_topk[k])
    struct_topk_frac = round(
        sum(supp_cat_all[c] for c in STRUCTURE_CATS) / max(sum(supp_cat_all.values()), 1), 4)

    # -- GATE B17 (pre-registered, audit-remediation plan P6-Step2) ----------------
    # COHERENCE MUST BE ENRICHMENT OVER THE VOCAB BASE-RATE, not raw suppression mass.
    # ~70% of the vocab is "content", so the raw structure-share of suppression is base-rate
    # dominated (uniform suppression already yields ~0.30) and would spuriously read as
    # "no structure". The B25-comparable signal is the TAIL: of the tokens the offset
    # suppresses HARDEST (top-k), what fraction is structure. base-rate ~0.30 => top-k >=0.66
    # (>=~2x) is a real structure signature (B25 scoring face got ~29-30/30 = ~0.97); top-k
    # near the base-rate is genuine flattening (no structure signature).
    n_struct = int(is_struct.sum()); n_cont = int(is_cont.sum())
    base_rate_struct = round(n_struct / max(n_struct + n_cont, 1), 4)
    enrichment_mass = round(coherence_frac / max(base_rate_struct, 1e-9), 3)      # mass-based
    enrichment_topk = round(struct_topk_frac / max(base_rate_struct, 1e-9), 3)    # tail (B25)
    COHERENT_TOPK_THR = 0.66     # top-k suppressed structure >= ~2x base-rate = B25-style signature
    GROW_THR = 0.05              # Q_last - Q_first materially positive
    coherent = struct_topk_frac >= COHERENT_TOPK_THR
    growing = slope["all"] >= GROW_THR
    flattening = struct_topk_frac <= base_rate_struct * 1.3   # tail barely above base-rate
    if coherent and growing:
        verdict = "COMMITMENT_SUPPORTED"           # B17 earns its number -> enters via ledger
    elif flattening and abs(slope["all"]) < GROW_THR:
        verdict = "FLATTENING_NO_STRUCTURE"        # B17 refuted (no structure signature)
    else:
        # structure signature present in the tail (top-k) but NOT decisively coherent-AND-growing
        verdict = "MIXED"                          # stays out of the draft (status quo)
    loop_vs_clean = round(traj["loop"][-1] - traj["clean_wrong"][-1], 4) if (
        q_frac_cnt["loop"][-1] and q_frac_cnt["clean_wrong"][-1]) else None

    payload = {
        "config": {"base": args.base, "offsets": args.offsets, "r6probe": args.r6probe,
                   "n_loops": args.n_loops, "n_cleanwrong": args.n_cleanwrong,
                   "max_new_tokens": args.max_new_tokens, "offset_scale": args.offset_scale,
                   "quartiles": nq, "topk": args.topk},
        "sanity_equivalence": {"nnsight_vs_hooks_cos_mean": cos_eq_mean, "per_item": cos_eq},
        "n_items": len(per_item),
        "gen_len_mean": {k: round(sum(v) / max(len(v), 1), 1) for k, v in gen_len_by_kind.items()},
        "structure_suppression_trajectory": traj,        # per quartile, per kind + all
        "slope_last_minus_first": slope,
        "coherence_structure_share": coherence_frac,     # mass-based, all positions
        "topk_suppressed_structure_frac": struct_topk_frac,   # B25-comparable
        "topk_suppressed_category_mix": {k: dict(supp_cat_topk[k].most_common()) for k in kinds},
        "loop_minus_clean_final_quartile": loop_vs_clean,
        "gate": {"coherent_topk_thr": COHERENT_TOPK_THR, "grow_thr": GROW_THR,
                 "base_rate_struct": base_rate_struct,
                 "enrichment_topk": enrichment_topk, "enrichment_mass": enrichment_mass,
                 "coherent": coherent, "growing": growing, "flattening": flattening,
                 "verdict": verdict,
                 "note": "coherent uses top-k structure enrichment over vocab base-rate (B25-"
                         "comparable), NOT raw mass share (base-rate dominated). enrichment_topk>=~2 "
                         "= real structure signature; ~1 = flattening."},
        "per_item": per_item,
        "read": "verdict COMMITMENT_SUPPORTED => structure-suppression is coherent (top-k structure "
                "enrichment >=~2x the vocab base-rate, i.e. struct_topk_frac>=0.66) AND grows along "
                "the generation (Q_last-Q_first>=0.05) => B17 commitment frame earns its number "
                "(enters via ledger). FLATTENING_NO_STRUCTURE => top-k near base-rate (no structure "
                "signature) => B17 refuted. MIXED => structure signature present in the tail but not "
                "decisively coherent-AND-growing => stays out of the draft. NOTE: mass_share is "
                "base-rate dominated (~70% of vocab is content) — do NOT gate on it. "
                "loop_minus_clean_final_quartile>0 => loops de-structure more than clean-wrong at the "
                "end (consistent with 'never commits to a close').",
        "method": "nnsight real-forward output-patch (r_layer @ o_proj.output == install_lofit_hooks) "
                  "over REAL adapter generation positions; structure/content via logit_lens categorize.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 78, flush=True)
    print(f"  trajectory (structure share by quartile):")
    for k in ("all", "loop", "clean_wrong"):
        print(f"    {k:12s}: {traj[k]}  slope={slope[k]:+.3f}", flush=True)
    print(f"  topk_struct_frac={struct_topk_frac} (enrich x{enrichment_topk} over base "
          f"{base_rate_struct}) | mass_share={coherence_frac} (enrich x{enrichment_mass}) | "
          f"loop-clean(Q_last)={loop_vs_clean}", flush=True)
    print(f"  SANITY cos={cos_eq_mean:.4f} | GATE VERDICT = {verdict}", flush=True)
    print(f"  saved: {args.out}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
