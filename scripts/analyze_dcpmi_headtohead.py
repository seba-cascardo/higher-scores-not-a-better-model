"""F0.2 — the EXPLOIT DECIDER: per-item head-to-head on ARC of the off-axis V7-mc
offset vs DC-PMI (a FREE baseline, Holtzman 2104.08315) on the mc_only set.

THE QUESTION
------------
The off-axis offset "fixes" 124 ARC items (the mc_only set: base wrong, mc right,
v_inf-offset wrong — acc_norm). DC-PMI (Domain-Conditional pointwise mutual
information) is a PARAMETER-FREE re-scoring that costs nothing on top of the same
logprobs:
    score_PMI(A_i) = logP(A_i | Q) - logP(A_i | C_neutro)
Does DC-PMI recover the SAME mc_only items the offset recovers?

  * OFF-AXIS-SPECIFIC set = items the offset fixes (gold->argmax) that DC-PMI does
    NOT fix, evaluated on the PMI-residual (i.e. DC-PMI argmax still wrong).
  * If that set is SMALL  -> the offset's recovery is mostly obtainable for FREE via
    PMI -> the EXPLOIT IS DEAD (the offset buys ~nothing PMI doesn't already give).
  * If that set is LARGE  -> the off-axis offset buys recovery PMI cannot -> ALIVE.

This is the cheap decider gate from the off-axis battery (F0.2) and the ClaraVT
action plan (DC-PMI as the free foil that decides "exploit vs espejismo"). It is
NOT the mechanism test; it only answers "is the recovery free?".

PRE-REGISTERED READ (encoded below, do NOT tune post-hoc)
---------------------------------------------------------
Let S = |mc_only| (items the offset fixes by definition) and
    K = |off-axis-specific| (offset fixes, DC-PMI does not, on the PMI-residual).
  * EXPLOIT ~DEAD  if  the paired-diff CI includes 0  OR  K <= OFFAXIS_SPECIFIC_DEAD_FRAC * S
                       (default 0.15, i.e. <= ~15% of mc_only).
  * EXPLOIT ALIVE  if  the paired-diff CI_lo > 0  AND  K >  OFFAXIS_SPECIFIC_DEAD_FRAC * S.
The fraction is read on the POINT estimate; SIGNIFICANCE is the PAIRED per-item
bootstrap of mean(offset_fix - pmi_base_fix) (== the off-axis-specific fraction,
since offset_fix is all-1 on mc_only), whose 2.5pct lower bound CAN fall to/below 0
when PMI fixes ~the same items -> a genuine test of off-axis-specificity vs PMI.
NOTE: the count-percentile CI (boot_count_ci) is DESCRIPTIVE ONLY — a bootstrap
percentile of a sample's own count is >= 0 by construction and so cannot test that
count against a null of 0; it is reported but NOT used in the verdict. Both DEAD
conditions are OR'd (either suffices) — the strictest read of "the offset buys nothing".

WHY acc_norm EVERYWHERE
-----------------------
The canonical +35 and the arbiter mc_only set are acc_norm (char-length-normalized
sum logprob). The offset/DC-PMI/answer-only argmaxes here ALL apply the SAME
char-len normalization (mirroring score_mc_pmi.py) so "recovered" is metric-
consistent with how mc_only was defined. The offset-arm recovery is recomputed
from d_mc and reconciled against the committed arbiter list (drift logged, not
silently trusted).

INPUTS (analysis-time; regenerated in Fase 1)
---------------------------------------------
  SAME-SCAFFOLD FIX: the PMI term logP(A|Q) - logP(A|C_neutro) needs BOTH terms scored
  under the SAME scaffold. base/mc conditional now comes from the merged premise file
  (same scaffold as the answer-only term), NOT from d_*.json (cross-scaffold lm-eval
  thinking-mode). d_null/d_mc are METADATA-ONLY here (option texts + gold).
  runs/vinf_causal/d_null.json    METADATA ONLY (option texts + gold; logprobs NOT used)
  runs/vinf_causal/d_mc.json      METADATA ONLY (option texts + gold; logprobs NOT used)
  runs/vinf_causal/d_vinf.json    v_inf conditional — re-derivation of mc_only ONLY (an
        internal argmax, not the PMI), so its scaffold does not enter the PMI subtraction.
  runs/vinf_causal/answer_only_logprobs.json   MERGED, self-sufficient:
        conditional.{base,mc}.<task>.<doc_id> -> [lp per option]  (same-scaffold logP(A|Q))
        arms.{base,mc}.<task>.<doc_id>        -> [lp per option]  (answer-only logP(A|C_neutro))
        Produced by score_mc_premise_pod.py + merge (run_phase1_pod.sh). See score_mc_pmi.py.
  runs/vinf_causal/pmi_dc_arc_challenge.json   summary PMI run (read for cross-check
        of the pooled PMI/naive accuracies; per-item is recomputed here).
  runs/vinf_causal/arbiter_arc_sets.json       strata; sets.mc_only = the 124 doc_ids.

OUTPUT
------
  print summary (mandatory per-iteration logging) + write
  runs/vinf_causal/dcpmi_headtohead_arc.json with: |mc_only|, recovered_by_offset
  (=all by def), recovered_by_pmi (overlap on mc_only), recovered_by_answeronly,
  the OFF-AXIS-SPECIFIC set (offset yes / PMI no) with size + bootstrap CI, the
  reverse (PMI yes / offset no, on the same comparison base), and the verdict.

CPU-local, EUR0.  Repo conventions: argparse, utf-8 stdout, flush prints, bootstrap
CI over items (mirrors score_mc_pmi.py / analyze_lift_surface_stratified.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows cp1252 console: do not break on arrows / delta / sub-indices.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- PRE-REGISTERED READ THRESHOLDS (do NOT tune post-hoc) -----------------------
# off-axis-specific K <= this fraction of |mc_only|  -> exploit ~dead (point read).
OFFAXIS_SPECIFIC_DEAD_FRAC = 0.15
# Significance guard: the PAIRED per-item bootstrap of mean(offset_fix - pmi_base_fix);
# if its 2.5pct CI includes 0 -> also dead, regardless of frac. (The count-percentile
# CI is reported but NOT used: a percentile of a sample's own count is >= 0 by
# construction and so cannot test the count against a null of 0.)


# ---------------------------------------------------------------------------------
# loaders / schema helpers (MIRROR score_mc_pmi.py so "recovered" is identical-by-
# construction with how mc_only and the +35 were defined).
# ---------------------------------------------------------------------------------
def load_task_samples(path: Path, task: str) -> dict[int, dict]:
    """doc_id -> sample, for one task of an lm-eval d_*.json (--log-samples)."""
    d = json.load(path.open(encoding="utf-8"))
    samples = d.get("samples", d)
    if task not in samples:
        raise KeyError(f"{path.name}: task '{task}' not present. Tasks: "
                       f"{list(samples.keys())[:8]}...")
    return {int(s["doc_id"]): s for s in samples[task]}


def option_texts(doc: dict) -> list[str]:
    """MC options, robust to ARC (choices.text), MMLU (choices list), TQA mc1
    (mc1_targets.choices). Option ORDER must match filtered_resps order."""
    ch = doc.get("choices")
    if isinstance(ch, dict) and "text" in ch:               # ARC / lm-eval multiple_choice
        return list(ch["text"])
    if isinstance(ch, list) and ch and isinstance(ch[0], str):  # MMLU
        return list(ch)
    mc1 = doc.get("mc1_targets")                            # truthfulqa_mc1
    if isinstance(mc1, dict) and "choices" in mc1:
        return list(mc1["choices"])
    raise KeyError(f"could not extract options from doc (keys: {list(doc.keys())})")


def cond_logprobs(sample: dict) -> np.ndarray:
    """logP(A_i | Q): raw SUM-token logprob per option from filtered_resps[i][0]."""
    return np.array([float(r[0]) for r in sample["filtered_resps"]], dtype=np.float64)


def char_lens(texts: list[str]) -> np.ndarray:
    return np.array([max(1, len(t)) for t in texts], dtype=np.float64)


def acc_norm_of(sample: dict) -> int | None:
    """0/1 acc_norm of an lm-eval sample (top-level or nested under metrics)."""
    a = sample.get("acc_norm")
    if a is None and isinstance(sample.get("metrics"), dict):
        a = sample["metrics"].get("acc_norm")
    if a is None and isinstance(sample.get("metrics"), dict):
        a = sample["metrics"].get("acc")  # last resort if acc_norm absent
    return int(round(float(a))) if a is not None else None


def argmax_correct(scores: np.ndarray, gold: int) -> int:
    """1 if argmax(scores)==gold else 0. Ties -> first index (np.argmax convention,
    same as lm-eval / score_mc_pmi.py)."""
    return int(int(np.argmax(scores)) == gold)


def boot_count_ci(mask: np.ndarray, n_boot: int, seed: int) -> tuple[int, float, float]:
    """Bootstrap CI for the COUNT of True entries in a per-item boolean mask.
    Resamples item indices with replacement; reports point count + 2.5/97.5 pct of
    the resampled counts (scaled to the original n)."""
    n = len(mask)
    if n == 0:
        return 0, float("nan"), float("nan")
    m = mask.astype(np.float64)
    rng = np.random.default_rng(seed)
    counts = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        counts[k] = float(m[idx].sum())  # expected count over a resample of size n
    return int(m.sum()), float(np.percentile(counts, 2.5)), float(np.percentile(counts, 97.5))


def boot_frac_ci(mask: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap CI for the FRACTION of True entries (mask.mean()) over items."""
    n = len(mask)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = mask.astype(np.float64)
    rng = np.random.default_rng(seed)
    fr = np.array([float(m[rng.integers(0, n, n)].mean()) for _ in range(n_boot)])
    return float(m.mean()), float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))


def boot_paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
                        ) -> tuple[float, float, float]:
    """Bootstrap CI for the mean PAIRED per-item difference mean(a - b) over items
    (a, b are per-item 0/1 fix flags on the SAME items). This is a REAL test of
    off-axis-specificity vs PMI: with a = offset_fix (==1 on all mc_only) and
    b = pmi_base_fix, mean(a - b) == the off-axis-specific FRACTION, and resampling
    items lets the lower CI bound fall to (or below) 0 when PMI fixes ~all the same
    items — unlike a percentile of the count itself, which is >= 0 by construction
    and so can never meaningfully include 0 for K > 0."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diff = a.astype(np.float64) - b.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([float(diff[rng.integers(0, n, n)].mean()) for _ in range(n_boot)])
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("runs/vinf_causal"))
    ap.add_argument("--task", default="arc_challenge",
                    help="lm-eval task (this gate is ARC; kept configurable for parity)")
    ap.add_argument("--null", type=Path, default=None, help="default <dir>/d_null.json")
    ap.add_argument("--mc", type=Path, default=None, help="default <dir>/d_mc.json")
    ap.add_argument("--vinf", type=Path, default=None, help="default <dir>/d_vinf.json")
    ap.add_argument("--answer-only", type=Path, default=None,
                    help="default <dir>/answer_only_logprobs.json (merged arms shape)")
    ap.add_argument("--pmi-summary", type=Path, default=None,
                    help="default <dir>/pmi_dc_<task>.json (read for pooled cross-check)")
    ap.add_argument("--arbiter", type=Path, default=None,
                    help="default <dir>/arbiter_arc_sets.json (sets.mc_only)")
    ap.add_argument("--metric", default="acc_norm", choices=["acc_norm", "acc"],
                    help="char-len normalization on/off; mc_only & +35 are acc_norm.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    use_norm = args.metric == "acc_norm"

    null_path = args.null or (args.dir / "d_null.json")
    mc_path = args.mc or (args.dir / "d_mc.json")
    vinf_path = args.vinf or (args.dir / "d_vinf.json")
    ao_path = args.answer_only or (args.dir / "answer_only_logprobs.json")
    pmi_summary_path = args.pmi_summary or (args.dir / f"pmi_dc_{args.task}.json")
    arbiter_path = args.arbiter or (args.dir / "arbiter_arc_sets.json")

    print("=" * 78, flush=True)
    print(f"F0.2 DC-PMI head-to-head (EXPLOIT DECIDER)  |  task={args.task}  "
          f"metric={args.metric}", flush=True)
    print("=" * 78, flush=True)
    print(f"  base (null)    : {null_path}", flush=True)
    print(f"  offset (mc)    : {mc_path}", flush=True)
    print(f"  v_inf          : {vinf_path}", flush=True)
    print(f"  answer-only    : {ao_path}", flush=True)
    print(f"  pmi summary    : {pmi_summary_path}", flush=True)
    print(f"  arbiter sets   : {arbiter_path}", flush=True)

    # --- load lm-eval conditional runs --------------------------------------------
    null = load_task_samples(null_path, args.task)
    mc = load_task_samples(mc_path, args.task)
    vinf = load_task_samples(vinf_path, args.task)

    # --- load answer-only (merged arms shape) -------------------------------------
    if not ao_path.exists():
        raise SystemExit(
            f"\n[FATAL] answer-only file not found: {ao_path}\n"
            f"        Regenerate via score_mc_premise_pod.py + merge (Fase 1). This "
            f"analysis is purely local once that file exists.")
    ao = json.load(ao_path.open(encoding="utf-8"))
    neutral_premise = ao.get("neutral_premise", "<unknown>")
    ao_arms = ao.get("arms", {})
    for arm in ("base", "mc"):
        if arm not in ao_arms or args.task not in ao_arms[arm]:
            raise SystemExit(f"[FATAL] answer-only file missing arms['{arm}']['{args.task}']")
    ao_base = ao_arms["base"][args.task]
    ao_mc = ao_arms["mc"][args.task]
    print(f"  neutral premise: {neutral_premise!r}", flush=True)

    # --- SAME-SCAFFOLD conditional logP(A|Q) (the fix) ----------------------------
    # The PMI term is logP(A|Q) - logP(A|C_neutro). Both must be scored under the SAME
    # scaffold or the subtraction is meaningless. The merged answer-only file carries
    # conditional.{base,mc}[task] scored under the SAME premise scaffold as the
    # answer-only term (score_mc_premise_pod.py). d_null.json / d_mc.json are lm-eval
    # thinking-mode runs (cross-scaffold) -> from here they are METADATA-ONLY (option
    # texts + gold), never the conditional logprobs that enter the PMI.
    cond_arms = ao.get("conditional", {})
    if not (isinstance(cond_arms.get("base"), dict) and cond_arms["base"].get(args.task)
            and isinstance(cond_arms.get("mc"), dict) and cond_arms["mc"].get(args.task)):
        raise SystemExit(
            f"[FATAL] answer-only file missing same-scaffold conditional "
            f"(expected conditional.base[{args.task!r}] and conditional.mc[{args.task!r}]). "
            f"This is the same-scaffold fix: regenerate via score_mc_premise_pod.py + merge "
            f"(run_phase1_pod.sh). The cross-scaffold d_*.json cond is NO longer used for PMI.")
    cond_base_ss = cond_arms["base"][args.task]   # {doc_id: [lp per option]} same-scaffold
    cond_mc_ss = cond_arms["mc"][args.task]
    print(f"  same-scaffold conditional: base={len(cond_base_ss)} mc={len(cond_mc_ss)}",
          flush=True)

    # --- load arbiter mc_only (committed strata) ----------------------------------
    # Optional: a committed arbiter exists only for ARC (it is a CoT artifact). For any
    # other task pass --arbiter none. Do NOT point a task at another task's arbiter:
    # doc_ids are 0..N-1 in every task, so the "reconciliation" below would intersect
    # unrelated items and silently pass.
    if str(arbiter_path).lower() == "none":
        mc_only_committed = None
        print("  committed mc_only: NONE (--arbiter none) -> mc_only is taken from the "
              "cold-scoring recomputation, which is its definition; no reconciliation.",
              flush=True)
    else:
        arb = json.load(arbiter_path.open(encoding="utf-8"))
        mc_only_committed = sorted(int(x) for x in arb["sets"]["mc_only"])
        print(f"  committed mc_only: {len(mc_only_committed)} doc_ids", flush=True)

    # --- optional: pooled PMI summary cross-check ---------------------------------
    pmi_summary = None
    if pmi_summary_path.exists():
        try:
            pmi_summary = json.load(pmi_summary_path.open(encoding="utf-8"))
            print(f"  pmi summary    : naive_lift={pmi_summary.get('naive',{}).get('lift')}  "
                  f"pmi_lift={pmi_summary.get('pmi',{}).get('lift')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] could not read pmi summary ({e}); continuing.", flush=True)
    else:
        print(f"  [INFO] pmi summary absent ({pmi_summary_path}); pooled cross-check "
              f"skipped (per-item is recomputed here regardless).", flush=True)

    # --- items in common across the four per-item sources --------------------------
    ids = sorted(set(null) & set(mc) & set(vinf)
                 & {int(k) for k in ao_base} & {int(k) for k in ao_mc}
                 & {int(k) for k in cond_base_ss} & {int(k) for k in cond_mc_ss})
    print(f"\n  aligned items  : {len(ids)} "
          f"(null={len(null)} mc={len(mc)} vinf={len(vinf)} "
          f"ao_base={len(ao_base)} ao_mc={len(ao_mc)} "
          f"cond_ss_base={len(cond_base_ss)} cond_ss_mc={len(cond_mc_ss)})", flush=True)
    if not ids:
        raise SystemExit("[FATAL] no items in common across the seven sources "
                         "(null/mc/vinf/ao_base/ao_mc/cond_base_ss/cond_mc_ss).")

    # --- per-item recompute: offset / DC-PMI / answer-only argmax correctness ------
    # Also re-derive the mc_only-defining correctness (base/mc/vinf acc_norm argmax)
    # from the SAME conditional logprobs so the head-to-head is self-consistent and
    # the committed arbiter list can be reconciled (drift logged, not trusted blind).
    rec = {}   # doc_id -> dict of 0/1 correctness flags
    n_opt_mismatch = 0
    opt_counts: list[int] = []
    for i, did in enumerate(ids):
        s_null, s_mc, s_vinf = null[did], mc[did], vinf[did]
        texts = option_texts(s_null["doc"])
        gold = int(s_null["target"])
        clen = char_lens(texts)
        nopt = len(texts)

        # SAME-SCAFFOLD conditional for base/mc (the fix): from the merged premise file,
        # NOT d_*.json (cross-scaffold lm-eval thinking-mode). v_inf has no premise-file
        # arm (score_mc_premise_pod runs base+mc only), so cond_v stays from d_vinf.json
        # and is used ONLY to re-derive mc_only (internal argmax) — never in the PMI
        # subtraction, so no cross-source mixing enters the PMI term.
        cond_b = np.array(cond_base_ss.get(str(did), []), dtype=np.float64)  # same-scaffold
        cond_m = np.array(cond_mc_ss.get(str(did), []), dtype=np.float64)    # same-scaffold
        cond_v = cond_logprobs(s_vinf)     # logP(A|Q) v_inf-offset (mc_only re-derivation only)
        ans_b = np.array(ao_base[str(did)], dtype=np.float64)  # logP(A|C_neutro) base
        ans_m = np.array(ao_mc[str(did)], dtype=np.float64)    # logP(A|C_neutro) mc

        # distinguish a same-scaffold conditional miss from a generic option mismatch
        if cond_b.size == 0 or cond_m.size == 0:
            print(f"  [WARN] item {did}: same-scaffold conditional missing "
                  f"(base={cond_b.size} mc={cond_m.size}); skip.", flush=True)
            n_opt_mismatch += 1
            continue
        if not (len(cond_b) == len(cond_m) == len(cond_v)
                == len(ans_b) == len(ans_m) == nopt):
            n_opt_mismatch += 1
            continue
        opt_counts.append(nopt)

        if use_norm:
            cond_b = cond_b / clen; cond_m = cond_m / clen; cond_v = cond_v / clen
            ans_b = ans_b / clen;   ans_m = ans_m / clen

        # mc_only-defining correctness (recomputed acc_norm argmax of the conditional):
        base_corr = argmax_correct(cond_b, gold)
        mc_corr = argmax_correct(cond_m, gold)      # == "recovered_by_offset" on the offset arm
        vinf_corr = argmax_correct(cond_v, gold)

        # DC-PMI on the OFFSET (mc) arm: logP(A|Q,mc) - logP(A|C_neutro,mc).
        # The exploit-decider question is "does the FREE PMI re-scoring (no offset
        # needed beyond what we already log) recover the same items?" — measured on
        # the BASE arm's PMI (PMI is parameter-free; the canonical free baseline uses
        # the base model's own logprobs). We compute BOTH and report base-PMI as the
        # primary free baseline; mc-PMI is reported for completeness.
        pmi_base_corr = argmax_correct(cond_b - ans_b, gold)
        pmi_mc_corr = argmax_correct(cond_m - ans_m, gold)
        # answer-only (choices-only) on the base arm:
        ao_base_corr = argmax_correct(ans_b, gold)

        rec[did] = dict(base=base_corr, offset=mc_corr, vinf=vinf_corr,
                        pmi_base=pmi_base_corr, pmi_mc=pmi_mc_corr,
                        answeronly_base=ao_base_corr)
        if (i + 1) % 100 == 0 or (i + 1) == len(ids):
            print(f"    [{i + 1}/{len(ids)}] scored", flush=True)

    if n_opt_mismatch:
        print(f"  [WARN] dropped {n_opt_mismatch} items for option-count mismatch "
              f"(answer-only vs conditional). Check option ordering in the pod pass.",
              flush=True)

    scored_ids = sorted(rec)
    chance = float(np.mean([1.0 / c for c in opt_counts])) if opt_counts else float("nan")

    # --- re-derive mc_only from recomputed correctness, reconcile with committed ---
    # mc_only := base wrong, offset(mc) right, v_inf wrong  (acc_norm; same as prep).
    mc_only_recomputed = sorted(
        did for did in scored_ids
        if rec[did]["base"] == 0 and rec[did]["offset"] == 1 and rec[did]["vinf"] == 0)
    set_recomputed = set(mc_only_recomputed)

    if mc_only_committed is None:
        # No committed arbiter for this task. mc_only IS the cold-scoring stratum, so
        # the recomputation is the definition, not a substitute for it. Nothing to
        # reconcile -- and we say so rather than printing a vacuous "matches exactly".
        print(f"\n  --- mc_only from recomputation: {len(set_recomputed)} doc_ids "
              f"(no committed arbiter exists for task '{args.task}'; not reconciled) ---",
              flush=True)
        mc_only_ids = sorted(set_recomputed)
    else:
        set_committed = set(mc_only_committed)
        only_committed = sorted(set_committed - set_recomputed)
        only_recomputed = sorted(set_recomputed - set_committed)
        print(f"\n  --- mc_only reconciliation (recomputed acc_norm vs committed arbiter) ---",
              flush=True)
        print(f"    committed={len(set_committed)}  recomputed={len(set_recomputed)}  "
              f"intersection={len(set_committed & set_recomputed)}", flush=True)
        if only_committed or only_recomputed:
            print(f"    [WARN] drift: in_committed_not_recomputed={len(only_committed)} "
                  f"{only_committed[:10]}{'...' if len(only_committed) > 10 else ''}  |  "
                  f"in_recomputed_not_committed={len(only_recomputed)} "
                  f"{only_recomputed[:10]}{'...' if len(only_recomputed) > 10 else ''}",
                  flush=True)
            print(f"    [WARN] proceeding on the RECOMPUTED-and-also-scored mc_only "
                  f"(self-consistent with the per-item PMI argmaxes).", flush=True)
        else:
            print(f"    OK: recomputed mc_only matches committed arbiter exactly.", flush=True)

        # Analysis base = mc_only items that we actually scored per-item (intersection of
        # committed and recomputed keeps us self-consistent AND faithful to the canonical
        # 124; fall back to recomputed if committed items were dropped for mismatch).
        mc_only_ids = sorted((set_committed & set_recomputed) or set_recomputed)
    S = len(mc_only_ids)
    print(f"\n  --- HEAD-TO-HEAD on mc_only (S={S}) ---", flush=True)
    if S == 0:
        raise SystemExit("[FATAL] mc_only analysis set is empty after reconciliation.")

    # Per-item flags on the mc_only set.
    offset_fix = np.array([rec[d]["offset"] for d in mc_only_ids], dtype=bool)   # all True by def
    pmi_base_fix = np.array([rec[d]["pmi_base"] for d in mc_only_ids], dtype=bool)
    pmi_mc_fix = np.array([rec[d]["pmi_mc"] for d in mc_only_ids], dtype=bool)
    ao_base_fix = np.array([rec[d]["answeronly_base"] for d in mc_only_ids], dtype=bool)

    # Primary free baseline = BASE-arm DC-PMI (parameter-free, no offset).
    recovered_by_offset = mc_only_ids[:]                       # all by def
    recovered_by_pmi = [d for d in mc_only_ids if rec[d]["pmi_base"] == 1]
    recovered_by_answeronly = [d for d in mc_only_ids if rec[d]["answeronly_base"] == 1]

    # OFF-AXIS-SPECIFIC = offset fixes (all of mc_only) AND DC-PMI(base) does NOT fix.
    offaxis_specific = [d for d in mc_only_ids if rec[d]["pmi_base"] == 0]
    offaxis_specific_mask = ~pmi_base_fix                      # offset always 1 on mc_only
    # REVERSE = DC-PMI(base) fixes AND offset does NOT — empty by construction on
    # mc_only (offset fixes all of mc_only). Reported on the BROADER scored set below
    # to give the honest reverse picture (where PMI buys what the offset does not).
    reverse_on_mc_only = []  # always empty on mc_only by definition; kept explicit.

    # sanity: offaxis_specific count == S - |recovered_by_pmi|
    assert len(offaxis_specific) == S - len(recovered_by_pmi)

    # --- bootstrap CIs over the mc_only items -------------------------------------
    k_off_point, k_off_lo, k_off_hi = boot_count_ci(offaxis_specific_mask, args.n_boot, args.seed)
    f_off_point, f_off_lo, f_off_hi = boot_frac_ci(offaxis_specific_mask, args.n_boot, args.seed)
    # REAL significance test of off-axis-specificity vs PMI: paired per-item diff
    # mean(offset_fix - pmi_base_fix). offset_fix is all-1 on mc_only, so this equals
    # the off-axis-specific fraction; resampling items lets the lower bound reach 0
    # when PMI fixes ~the same items. Gate the ALIVE verdict on its 2.5pct > 0.
    d_off_point, d_off_lo, d_off_hi = boot_paired_diff_ci(
        offset_fix, pmi_base_fix, args.n_boot, args.seed)
    # overlap (PMI recovers) fraction CI, for context.
    f_pmi_point, f_pmi_lo, f_pmi_hi = boot_frac_ci(pmi_base_fix, args.n_boot, args.seed)
    f_ao_point, f_ao_lo, f_ao_hi = boot_frac_ci(ao_base_fix, args.n_boot, args.seed)
    f_pmimc_point, f_pmimc_lo, f_pmimc_hi = boot_frac_ci(pmi_mc_fix, args.n_boot, args.seed)

    K = len(offaxis_specific)  # off-axis-specific point count
    print(f"    |mc_only| (S)                 = {S}", flush=True)
    print(f"    recovered_by_offset           = {len(recovered_by_offset)} (=S by def)", flush=True)
    print(f"    recovered_by_pmi (base, free) = {len(recovered_by_pmi)}  "
          f"frac={f_pmi_point:.3f} CI[{f_pmi_lo:.3f},{f_pmi_hi:.3f}]", flush=True)
    print(f"    recovered_by_answeronly(base) = {len(recovered_by_answeronly)}  "
          f"frac={f_ao_point:.3f} CI[{f_ao_lo:.3f},{f_ao_hi:.3f}]  (chance~{chance:.3f})",
          flush=True)
    print(f"    recovered_by_pmi (mc arm)     = {int(pmi_mc_fix.sum())}  "
          f"frac={f_pmimc_point:.3f} CI[{f_pmimc_lo:.3f},{f_pmimc_hi:.3f}]  (context)",
          flush=True)
    print(f"    OFF-AXIS-SPECIFIC (K)         = {K}  "
          f"count CI[{k_off_lo:.1f},{k_off_hi:.1f}] (descriptive; NOT a sig test)  "
          f"frac={f_off_point:.3f} CI[{f_off_lo:.3f},{f_off_hi:.3f}]", flush=True)
    print(f"    paired diff (offset-pmi)      = {d_off_point:+.3f}  "
          f"CI[{d_off_lo:+.3f},{d_off_hi:+.3f}]  (REAL sig test of off-axis vs PMI)",
          flush=True)
    print(f"    reverse (pmi yes / offset no) = {len(reverse_on_mc_only)} "
          f"(empty by construction on mc_only)", flush=True)

    # --- broader reverse picture: on the FULL scored set, where does base-PMI fix an
    # item the offset does not? (honest "what PMI buys that the offset misses"). ----
    reverse_full = sorted(d for d in scored_ids
                          if rec[d]["base"] == 0 and rec[d]["pmi_base"] == 1
                          and rec[d]["offset"] == 0)
    print(f"    reverse (full scored set: base-wrong, pmi-fixes, offset-misses) = "
          f"{len(reverse_full)}", flush=True)

    # --- PRE-REGISTERED VERDICT ---------------------------------------------------
    # Significance is the PAIRED-DIFFERENCE test mean(offset_fix - pmi_base_fix): its
    # 2.5pct lower bound CAN fall to/below 0 when PMI fixes ~the same items, so it is a
    # genuine test of off-axis-specificity vs PMI. (The old count-percentile CI was
    # inert: a bootstrap percentile of a sample's own count is >= 0 by construction and
    # only "includes 0" when K is tiny, so it could never test K against a null of 0.)
    dead_frac_thresh = OFFAXIS_SPECIFIC_DEAD_FRAC * S
    diff_ci_includes_zero = (d_off_lo <= 0.0)
    below_frac = (K <= dead_frac_thresh)
    if diff_ci_includes_zero or below_frac:
        reasons = []
        if diff_ci_includes_zero:
            reasons.append(f"paired-diff (offset-pmi) CI[{d_off_lo:+.3f},{d_off_hi:+.3f}] "
                           f"includes 0 (off-axis-specificity not sig vs PMI)")
        if below_frac:
            reasons.append(f"K={K} <= {OFFAXIS_SPECIFIC_DEAD_FRAC:.0%} of S "
                           f"({dead_frac_thresh:.1f})")
        verdict = "EXPLOIT ~DEAD"
        detail = ("the offset's mc_only recovery is mostly FREE via DC-PMI "
                  f"({len(recovered_by_pmi)}/{S} of the fixed items are already fixed by "
                  f"the parameter-free PMI re-scoring); " + " AND ".join(reasons))
        alive = False
    else:
        verdict = "EXPLOIT ALIVE"
        detail = (f"the off-axis offset recovers K={K} mc_only items that DC-PMI does NOT "
                  f"(> {OFFAXIS_SPECIFIC_DEAD_FRAC:.0%} of S and paired-diff CI_lo="
                  f"{d_off_lo:+.3f} > 0) -> the offset buys recovery the free PMI "
                  f"baseline cannot.")
        alive = True

    print(f"\n  --- PRE-REGISTERED VERDICT ---", flush=True)
    print(f"    thresholds: dead if  paired_diff(offset-pmi)_CI_lo <= 0  OR  "
          f"K <= {OFFAXIS_SPECIFIC_DEAD_FRAC:.2f}*S ({dead_frac_thresh:.1f})", flush=True)
    print(f"    [{verdict}]  {detail}", flush=True)

    # --- persist ------------------------------------------------------------------
    payload = dict(
        gate="F0.2_dcpmi_headtohead_exploit_decider",
        task=args.task,
        metric=args.metric,
        neutral_premise=neutral_premise,
        chance=chance,
        n_scored=len(scored_ids),
        n_opt_mismatch=n_opt_mismatch,
        thresholds=dict(
            offaxis_specific_dead_frac=OFFAXIS_SPECIFIC_DEAD_FRAC,
            dead_frac_count=dead_frac_thresh,
            rule=("EXPLOIT_DEAD if (paired_diff(offset-pmi)_CI_lo <= 0) OR (K <= "
                  "OFFAXIS_SPECIFIC_DEAD_FRAC * |mc_only|); else EXPLOIT_ALIVE. "
                  "Significance = paired per-item bootstrap of mean(offset_fix - "
                  "pmi_base_fix); the count-percentile CI is descriptive only "
                  "(>= 0 by construction, not a test against a null of 0)."),
        ),
        mc_only=dict(
            committed=len(mc_only_committed),
            recomputed=len(mc_only_recomputed),
            analysis_S=S,
            drift_in_committed_not_recomputed=only_committed,
            drift_in_recomputed_not_committed=only_recomputed,
            ids=mc_only_ids,
        ),
        recovered_by_offset=dict(n=len(recovered_by_offset), ids=recovered_by_offset),
        recovered_by_pmi_base=dict(
            n=len(recovered_by_pmi), ids=recovered_by_pmi,
            frac=f_pmi_point, frac_ci=[f_pmi_lo, f_pmi_hi]),
        recovered_by_answeronly_base=dict(
            n=len(recovered_by_answeronly), ids=recovered_by_answeronly,
            frac=f_ao_point, frac_ci=[f_ao_lo, f_ao_hi]),
        recovered_by_pmi_mc=dict(
            n=int(pmi_mc_fix.sum()), frac=f_pmimc_point, frac_ci=[f_pmimc_lo, f_pmimc_hi]),
        offaxis_specific=dict(
            n=K, ids=offaxis_specific,
            count_ci=[k_off_lo, k_off_hi],          # descriptive only (>= 0 by construction)
            frac_of_mc_only=f_off_point, frac_ci=[f_off_lo, f_off_hi],
            paired_diff_offset_minus_pmi=d_off_point,   # REAL sig test of off-axis vs PMI
            paired_diff_ci=[d_off_lo, d_off_hi]),
        reverse_pmi_yes_offset_no=dict(
            on_mc_only=len(reverse_on_mc_only),  # 0 by construction
            on_full_scored_set=dict(n=len(reverse_full), ids=reverse_full)),
        pmi_summary_crosscheck=(dict(
            naive_lift=pmi_summary.get("naive", {}).get("lift"),
            pmi_lift=pmi_summary.get("pmi", {}).get("lift"),
            pmi_frac_of_naive=pmi_summary.get("pmi", {}).get("pmi_frac_of_naive"),
        ) if pmi_summary else None),
        verdict=dict(label=verdict, alive=alive, detail=detail),
    )
    # Task-derived, never hardcoded: the original default wrote every task to
    # "dcpmi_headtohead_arc.json", so the TQA run landed under an ARC name and its
    # numbers looked artifact-less for months (see B62-TRAZA).
    out = args.out or (args.dir / f"dcpmi_headtohead_{args.task}.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {out}", flush=True)


if __name__ == "__main__":
    main()
