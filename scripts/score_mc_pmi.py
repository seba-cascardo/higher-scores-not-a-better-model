"""PMI-DC / choices-only test for the +35 MC lift (Holtzman surface-form competition;
Balepur/Chandak choices-only).  Does the contrastive-correctness adapter (mc) win MC by
recovering question-conditional KNOWLEDGE, or by exploiting the surface-form / answer-prior
of the options (a metric game off the correctness axis)?

Two diagnostics, both PRE-REGISTERED with kill-rules:

  (1) PMI-DC (Holtzman 2021 "Surface Form Competition"; domain-conditional PMI).
      Re-score every option by its POINTWISE MUTUAL INFORMATION with the question:
          score_PMI(A_i) = logP(A_i | Q) - logP(A_i | C_neutro)
      where C_neutro is a neutral/question-ablated premise (e.g. "Answer:" with the
      question text removed). The conditional logP(A_i|Q) ALREADY EXISTS in the
      d_*.json runs (filtered_resps[i][0]); the answer-only term logP(A_i|C_neutro)
      is the MISSING piece produced by the POD pass (see --answer-only).
      If the naive (conditional-only) lift is mostly surface prior, PMI normalization
      removes the answer-prior bias and the lift should COLLAPSE.
        KILL-RULE A (SURFACE-FORM):  Lift_PMI < 0.30 * Lift_naive  ->  the +35 is
        surface-form competition, NOT question-conditional knowledge.

  (2) Choices-only (Balepur 2024 "Artifacts or Abduction"; Chandak 2024).
      Score each option by its ANSWER-ONLY logprob alone (logP(A_i|C_neutro)),
      argmax -> "choices-only accuracy" (the model picks an answer WITHOUT seeing
      the question). If the mc adapter lifts choices-only accuracy meaningfully over
      base, the adapter is sharpening an ANSWER-PRIOR (which option "looks like an
      answer") rather than question-conditional reasoning.
        KILL-RULE B (ANSWER-PRIOR):  acc_choicesonly(mc) - acc_choicesonly(base)
        >= MARGIN (default 0.05, i.e. >=1/3 of a +0.15-scale real effect, and CI_lo>0)
        ->  the lift is (partly) an answer-prior, not question-conditional knowledge.

This script is the LOCAL "combine" stage: it reads the existing conditional runs
(d_null.json = base, d_mc.json = adapter) AND a provided answer-only-logprobs file
(produced on the pod, per option, per arm), then computes acc_naive / acc_PMI /
acc_choicesonly for base and mc, the lifts, bootstrap CIs over items, and applies
the kill-rules.  CPU-local, EUR0.

------------------------------------------------------------------------------------
INPUTS
------------------------------------------------------------------------------------
  --null   runs/vinf_causal/d_null.json   base conditional run (lm-eval --log-samples)
  --mc     runs/vinf_causal/d_mc.json     adapter conditional run (the "+35")
  --answer-only  runs/vinf_causal/answer_only_logprobs.json   POD output (see schema)

ANSWER-ONLY FILE SCHEMA (what the pod pass must emit; see scripts/score_mc_pmi.py
docstring section B and run_lm_eval_v7 recommendation in the handoff):

  {
    "neutral_premise": "Answer:",          # the C_neutro string used (audit)
    "arms": {
      "base": {                            # base model, NO adapter
        "<task>": {                        # e.g. "arc_challenge", "truthfulqa_mc1"
          "<doc_id>": [lp_A0, lp_A1, ...]  # logP(A_i | C_neutro), one per option,
                                           # SAME option order as filtered_resps
        }
      },
      "mc": { ... same shape ... }         # adapter arm
    }
  }

  Per-option logprobs MUST be the raw SUM token logprob of the continuation string
  (NOT length-normalized) so that PMI = logP(A|Q) - logP(A|C_neutro) is a clean
  difference of comparable quantities. The 'combine' mode applies acc_norm
  (char-length normalization) consistently to BOTH terms when --metric acc_norm,
  mirroring the canonical +35 (which is acc_norm).

------------------------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------------------------
  print summary + runs/vinf_causal/pmi_dc_<task>.json with per-arm accuracies,
  lifts, bootstrap CIs, and verdicts (SURFACE-FORM / ANSWER-PRIOR / KNOWLEDGE-CONSISTENT).

Repo conventions: argparse, sys.stdout.reconfigure utf-8, progress prints with flush,
JSON save, bootstrap CIs over items (matches analyze_lift_surface_stratified.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows cp1252 console: que no rompa con flechas/delta/sub-indices.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- pre-registered kill-rule thresholds (do NOT tune post-hoc) -----------------
PMI_SURFACE_FRAC = 0.30   # Lift_PMI < 0.30*Lift_naive -> SURFACE-FORM (kill-rule A)
ANSWER_PRIOR_MARGIN = 0.05  # acc_choicesonly(mc)-base >= this AND CI_lo>0 -> ANSWER-PRIOR (kill-rule B)


# ---------------------------------------------------------------------------------
# loaders / schema helpers (mirror analyze_lift_surface_stratified.py)
# ---------------------------------------------------------------------------------
def load_task_samples(path: Path, task: str) -> dict[int, dict]:
    """doc_id -> sample, for one task of an lm-eval d_*.json (--log-samples)."""
    d = json.load(path.open(encoding="utf-8"))
    if task not in d["samples"]:
        raise KeyError(f"{path.name}: task '{task}' not present. Tasks: "
                       f"{list(d['samples'].keys())[:8]}...")
    return {int(s["doc_id"]): s for s in d["samples"][task]}


def option_texts(doc: dict) -> list[str]:
    """MC options, robust to ARC (choices.text), MMLU (choices list) and TQA mc1
    (mc1_targets.choices). Option ORDER must match filtered_resps order."""
    ch = doc.get("choices")
    if isinstance(ch, dict) and "text" in ch:          # ARC / lm-eval multiple_choice
        return list(ch["text"])
    if isinstance(ch, list) and ch and isinstance(ch[0], str):  # MMLU
        return list(ch)
    mc1 = doc.get("mc1_targets")                        # truthfulqa_mc1
    if isinstance(mc1, dict) and "choices" in mc1:
        return list(mc1["choices"])
    raise KeyError(f"could not extract options from doc (keys: {list(doc.keys())})")


def cond_logprobs(sample: dict) -> np.ndarray:
    """logP(A_i | Q): the raw SUM-token logprob per option from filtered_resps."""
    return np.array([float(r[0]) for r in sample["filtered_resps"]], dtype=np.float64)


def char_lens(texts: list[str]) -> np.ndarray:
    return np.array([max(1, len(t)) for t in texts], dtype=np.float64)


def boot_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b) over paired items (a,b are per-item 0/1 correct).
    Resamples item indices jointly so the pairing is preserved."""
    n = len(a)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = float(a[idx].mean() - b[idx].mean())
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def boot_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    n = len(x)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = np.array([float(x[rng.integers(0, n, n)].mean()) for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def argmax_correct(scores: np.ndarray, gold: int) -> int:
    """1 if argmax(scores)==gold else 0. Ties -> first index (np.argmax convention,
    same as lm-eval)."""
    return int(int(np.argmax(scores)) == gold)


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("runs/vinf_causal"))
    ap.add_argument("--task", default="arc_challenge",
                    help="lm-eval task name (arc_challenge | truthfulqa_mc1 | ...)")
    ap.add_argument("--null", type=Path, default=None,
                    help="base conditional run (default <dir>/d_null.json)")
    ap.add_argument("--mc", type=Path, default=None,
                    help="adapter conditional run (default <dir>/d_mc.json)")
    ap.add_argument("--answer-only", type=Path, default=None,
                    help="POD answer-only logprobs file (default "
                         "<dir>/answer_only_logprobs.json). See module docstring for schema.")
    ap.add_argument("--premise-file", type=Path, default=None,
                    help="SELF-CONSISTENT mode: a per-task file from score_mc_premise_pod.py "
                         "(carries cond + answer-only + gold + charlen for both arms under the "
                         "SAME scaffold). When set, PMI uses that file's own conditional (clean "
                         "within-scaffold difference) and d_*.json is NOT read.")
    ap.add_argument("--metric", default="acc_norm", choices=["acc_norm", "acc"],
                    help="canonical lift metric (the +35 is acc_norm). Applies char-len "
                         "normalization consistently to BOTH PMI terms when acc_norm.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    use_norm = args.metric == "acc_norm"

    base_naive, mc_naive = [], []          # argmax logP(A|Q)              (= canonical, sanity)
    base_pmi, mc_pmi = [], []              # argmax [logP(A|Q)-logP(A|Cn)] (PMI-DC)
    base_co, mc_co = [], []                # argmax logP(A|Cn)             (choices-only)
    n_opt_mismatch = 0
    opt_counts: list[int] = []

    if args.premise_file is not None:
        # SELF-CONSISTENT: pod per-task file (score_mc_premise_pod.py) carries cond +
        # answer-only + gold + charlen for both arms under the SAME scaffold -> PMI is a
        # clean within-scaffold difference (no mixing with the lm-eval d_*.json conditional).
        print("=" * 74, flush=True)
        print(f"PMI-DC / choices-only [SELF-CONSISTENT premise-file]  |  task={args.task}  "
              f"metric={args.metric}", flush=True)
        print("=" * 74, flush=True)
        pf = json.load(args.premise_file.open(encoding="utf-8"))
        if pf.get("task") and pf["task"] != args.task:
            print(f"  [WARN] premise-file task {pf['task']!r} != --task {args.task!r}", flush=True)
        neutral_premise = pf.get("neutral_premise", "<unknown>")
        cb_all, cm_all = pf["conditional"]["base"], pf["conditional"]["mc"]
        ab_all, am_all = pf["answer_only"]["base"], pf["answer_only"]["mc"]
        gold_all, clen_all = pf["gold"], pf["charlen"]
        print(f"  premise file    : {args.premise_file}", flush=True)
        print(f"  neutral premise : {neutral_premise!r}   "
              f"sanity cond_lift={pf.get('sanity', {}).get('cond_lift')}", flush=True)
        ids = sorted(set(int(k) for k in cb_all) & set(int(k) for k in cm_all)
                     & set(int(k) for k in ab_all) & set(int(k) for k in am_all))
        print(f"  aligned items   : {len(ids)}", flush=True)
        for did in ids:
            k = str(did)
            cb = np.array(cb_all[k], dtype=np.float64); cm = np.array(cm_all[k], dtype=np.float64)
            ab = np.array(ab_all[k], dtype=np.float64); am = np.array(am_all[k], dtype=np.float64)
            gold = int(gold_all[k]); clen = np.array(clen_all[k], dtype=np.float64)
            nopt = len(clen)
            if not (len(cb) == len(cm) == len(ab) == len(am) == nopt):
                n_opt_mismatch += 1
                continue
            opt_counts.append(nopt)
            if use_norm:
                cb, cm, ab, am = cb / clen, cm / clen, ab / clen, am / clen
            base_naive.append(argmax_correct(cb, gold)); mc_naive.append(argmax_correct(cm, gold))
            base_pmi.append(argmax_correct(cb - ab, gold)); mc_pmi.append(argmax_correct(cm - am, gold))
            base_co.append(argmax_correct(ab, gold)); mc_co.append(argmax_correct(am, gold))
    else:
        null_path = args.null or (args.dir / "d_null.json")
        mc_path = args.mc or (args.dir / "d_mc.json")
        ao_path = args.answer_only or (args.dir / "answer_only_logprobs.json")

        print("=" * 74, flush=True)
        print(f"PMI-DC / choices-only  |  task={args.task}  metric={args.metric}", flush=True)
        print("=" * 74, flush=True)
        print(f"  conditional base : {null_path}", flush=True)
        print(f"  conditional mc   : {mc_path}", flush=True)
        print(f"  answer-only file : {ao_path}", flush=True)

        null = load_task_samples(null_path, args.task)
        mc = load_task_samples(mc_path, args.task)

        if not ao_path.exists():
            raise SystemExit(
                f"\n[FATAL] answer-only file not found: {ao_path}\n"
                f"        Run the POD pass (score_mc_premise_pod.py) or pass --premise-file. "
                f"This 'combine' stage is purely local once that file exists.")
        ao = json.load(ao_path.open(encoding="utf-8"))
        neutral_premise = ao.get("neutral_premise", "<unknown>")
        ao_arms = ao["arms"]
        for arm in ("base", "mc"):
            if arm not in ao_arms or args.task not in ao_arms[arm]:
                raise SystemExit(f"[FATAL] answer-only file missing arms['{arm}']['{args.task}']")
        ao_base = ao_arms["base"][args.task]
        ao_mc = ao_arms["mc"][args.task]
        print(f"  neutral premise  : {neutral_premise!r}", flush=True)

        ids = sorted(set(null) & set(mc) & {int(k) for k in ao_base} & {int(k) for k in ao_mc})
        print(f"  aligned items    : {len(ids)} "
              f"(null={len(null)} mc={len(mc)} ao_base={len(ao_base)} ao_mc={len(ao_mc)})",
              flush=True)
        if not ids:
            raise SystemExit("[FATAL] no items in common across the four sources.")

        for i, did in enumerate(ids):
            s_null, s_mc = null[did], mc[did]
            texts = option_texts(s_null["doc"])
            gold = int(s_null["target"])
            clen = char_lens(texts)
            cond_b = cond_logprobs(s_null); cond_m = cond_logprobs(s_mc)
            ans_b = np.array(ao_base[str(did)], dtype=np.float64)
            ans_m = np.array(ao_mc[str(did)], dtype=np.float64)
            nopt = len(texts)
            if not (len(cond_b) == len(cond_m) == len(ans_b) == len(ans_m) == nopt):
                n_opt_mismatch += 1
                continue
            opt_counts.append(nopt)
            if use_norm:
                cond_b = cond_b / clen; cond_m = cond_m / clen
                ans_b = ans_b / clen; ans_m = ans_m / clen
            base_naive.append(argmax_correct(cond_b, gold)); mc_naive.append(argmax_correct(cond_m, gold))
            base_pmi.append(argmax_correct(cond_b - ans_b, gold)); mc_pmi.append(argmax_correct(cond_m - ans_m, gold))
            base_co.append(argmax_correct(ans_b, gold)); mc_co.append(argmax_correct(ans_m, gold))
            if (i + 1) % 100 == 0 or (i + 1) == len(ids):
                print(f"    [{i + 1}/{len(ids)}] scored", flush=True)

    if n_opt_mismatch:
        print(f"  [WARN] dropped {n_opt_mismatch} items for option-count mismatch "
              f"(answer-only vs conditional). Check option ordering in the pod pass.", flush=True)

    base_naive = np.array(base_naive); mc_naive = np.array(mc_naive)
    base_pmi = np.array(base_pmi);     mc_pmi = np.array(mc_pmi)
    base_co = np.array(base_co);       mc_co = np.array(mc_co)
    n = len(base_naive)
    chance = float(np.mean([1.0 / c for c in opt_counts])) if opt_counts else float("nan")

    # --- accuracies + lifts (paired bootstrap over items) --------------------------
    def acc(x):  # point accuracy
        return float(x.mean())

    lift_naive, ln_lo, ln_hi = boot_diff_ci(mc_naive, base_naive, args.n_boot, args.seed)
    lift_pmi, lp_lo, lp_hi = boot_diff_ci(mc_pmi, base_pmi, args.n_boot, args.seed)
    lift_co, lc_lo, lc_hi = boot_diff_ci(mc_co, base_co, args.n_boot, args.seed)

    # choices-only ABSOLUTE accuracies (for the "near random?" read) + their CIs
    co_b_m, co_b_lo, co_b_hi = boot_mean_ci(base_co, args.n_boot, args.seed)
    co_m_m, co_m_lo, co_m_hi = boot_mean_ci(mc_co, args.n_boot, args.seed)

    # chance baseline (mean 1/n_opt over kept items) was computed during item scoring.

    print(f"\n  --- SANITY (naive argmax logP(A|Q), should reproduce the +35) ---", flush=True)
    print(f"    base_naive={acc(base_naive):.4f}  mc_naive={acc(mc_naive):.4f}  "
          f"lift_naive={lift_naive:+.4f}  CI[{ln_lo:+.4f},{ln_hi:+.4f}]", flush=True)

    print(f"\n  --- (1) PMI-DC  score = logP(A|Q) - logP(A|C_neutro) ---", flush=True)
    print(f"    base_PMI  ={acc(base_pmi):.4f}  mc_PMI  ={acc(mc_pmi):.4f}  "
          f"lift_PMI ={lift_pmi:+.4f}  CI[{lp_lo:+.4f},{lp_hi:+.4f}]", flush=True)

    print(f"\n  --- (2) Choices-only  score = logP(A|C_neutro) ---  (chance~{chance:.3f})", flush=True)
    print(f"    base_CO   ={co_b_m:.4f} CI[{co_b_lo:.4f},{co_b_hi:.4f}]  "
          f"mc_CO   ={co_m_m:.4f} CI[{co_m_lo:.4f},{co_m_hi:.4f}]", flush=True)
    print(f"    lift_CO  ={lift_co:+.4f}  CI[{lc_lo:+.4f},{lc_hi:+.4f}]", flush=True)

    # --- PRE-REGISTERED KILL-RULES -------------------------------------------------
    # A: SURFACE-FORM. Guard the ratio: only meaningful if the naive lift is positive
    #    and non-trivial. If lift_naive <= 0 the test is undefined (report N/A).
    pmi_frac = (lift_pmi / lift_naive) if abs(lift_naive) > 1e-9 else float("nan")
    if not (lift_naive > 0.02):
        verdict_A = "N/A (naive lift not positive/non-trivial; PMI ratio undefined)"
        surface_form = None
    elif lift_pmi < PMI_SURFACE_FRAC * lift_naive:
        verdict_A = (f"SURFACE-FORM  (lift_PMI {lift_pmi:+.4f} < {PMI_SURFACE_FRAC:.2f}*"
                     f"lift_naive {PMI_SURFACE_FRAC * lift_naive:+.4f}; PMI retains only "
                     f"{pmi_frac:.0%} of the naive lift)")
        surface_form = True
    else:
        verdict_A = (f"NOT surface-form  (PMI retains {pmi_frac:.0%} of the naive lift, "
                     f">= {PMI_SURFACE_FRAC:.0%} threshold)")
        surface_form = False

    # B: ANSWER-PRIOR. mc lifts choices-only acc over base by >= MARGIN AND CI_lo>0.
    if lift_co >= ANSWER_PRIOR_MARGIN and lc_lo > 0:
        verdict_B = (f"ANSWER-PRIOR  (mc lifts choices-only acc by {lift_co:+.4f} "
                     f">= {ANSWER_PRIOR_MARGIN:.2f} and CI_lo {lc_lo:+.4f} > 0 -> the adapter "
                     f"sharpens which option 'looks like an answer', question-independent)")
        answer_prior = True
    else:
        verdict_B = (f"no answer-prior lift  (choices-only lift {lift_co:+.4f} "
                     f"CI[{lc_lo:+.4f},{lc_hi:+.4f}] does not clear "
                     f">= {ANSWER_PRIOR_MARGIN:.2f} with CI_lo>0)")
        answer_prior = False

    # Combined read.
    if surface_form or answer_prior:
        overall = "LIFT IS (PARTLY) SURFACE-FORM / ANSWER-PRIOR, not question-conditional knowledge"
    elif surface_form is False and not answer_prior:
        overall = ("KNOWLEDGE-CONSISTENT: PMI survives and choices-only does not lift -> the "
                   "+35 is question-conditional under this test (does NOT by itself rule out the "
                   "off-axis OOD-hack read; complementary to the v_inf causal result)")
    else:
        overall = "INCONCLUSIVE (see per-rule verdicts)"

    print(f"\n  --- VERDICTS (pre-registered) ---", flush=True)
    print(f"    [A surface-form] {verdict_A}", flush=True)
    print(f"    [B answer-prior] {verdict_B}", flush=True)
    print(f"    [overall]        {overall}", flush=True)

    payload = dict(
        task=args.task, metric=args.metric, n=n, neutral_premise=neutral_premise,
        chance=chance,
        thresholds=dict(pmi_surface_frac=PMI_SURFACE_FRAC,
                        answer_prior_margin=ANSWER_PRIOR_MARGIN),
        naive=dict(base=acc(base_naive), mc=acc(mc_naive),
                   lift=lift_naive, ci=[ln_lo, ln_hi]),
        pmi=dict(base=acc(base_pmi), mc=acc(mc_pmi),
                 lift=lift_pmi, ci=[lp_lo, lp_hi], pmi_frac_of_naive=pmi_frac),
        choices_only=dict(base=co_b_m, base_ci=[co_b_lo, co_b_hi],
                          mc=co_m_m, mc_ci=[co_m_lo, co_m_hi],
                          lift=lift_co, lift_ci=[lc_lo, lc_hi]),
        verdicts=dict(surface_form=surface_form, answer_prior=answer_prior,
                      A=verdict_A, B=verdict_B, overall=overall),
    )
    out = args.out or (args.dir / f"pmi_dc_{args.task}.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {out}", flush=True)


if __name__ == "__main__":
    main()
