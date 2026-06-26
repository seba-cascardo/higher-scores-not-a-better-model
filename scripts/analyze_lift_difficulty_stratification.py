"""Difficulty / burial stratification of the cold-MC lift (ClaraVT, 2026-06-26).

The user's hypothesis: the adapter is a fast System-1 readout that rescues answers
the model already knows easily, and "does NOT replicate on hard questions". We never
measured that. This is the EUR0-local first cut, from the existing per-item logprob
runs (d_null.json = base, d_mc.json = adapter) plus the gen-CoT arbiter
(arbiter_arc_cot.json, which ran CoT on ALL 400 items -> base_cot_correct per item).

Two questions, both restricted to BASE-WRONG items (b_corr==0), metric = acc_norm:

  Q1 (BURIAL gradient): does the adapter's fix-rate fall as the gold is more buried
     in the base? Bin by base gold-rank (1/2/3+) and by the chosen-vs-gold z-gap.
     ClaraVT's prediction: fix-rate falls with burial -> rescues near-surface golds.
     CONFOUND (Metodologo): a fixed-magnitude promotion ALSO fixes small gaps and
     misses big ones, so a falling gradient could be trivial gap-closing.

  Q2 (RECOVERABILITY, the de-confounded test): does CoT-accessibility predict the
     fix beyond burial? 2x2 of fix x base_cot_correct among base-wrong items. The
     load-bearing cell is base-wrong AND CoT-WRONG (genuinely hard for this model):
     if the adapter rarely fixes those -> "shortcut to recoverable knowledge"; if it
     fixes them often -> surface-form magic that does not need the model to know.
     POWER CAVEAT: on ARC the base solves 387/400 with CoT, so the CoT-wrong cell is
     ~13 items -> underpowered. That scarcity is itself the argument for measuring the
     cold-MC lift on a genuinely-hard benchmark (GPQA, options-present) on the pod.

Repo conventions: argparse, utf-8 stdout, progress prints with flush, JSON save,
paired bootstrap CIs over items. CPU-local, numpy-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------------
# loaders (mirror analyze_cold_mc_calibration.py / score_mc_pmi.py)
# ---------------------------------------------------------------------------------
def load_task_samples(path: Path, task: str) -> dict[int, dict]:
    d = json.load(path.open(encoding="utf-8"))
    if task not in d["samples"]:
        raise KeyError(f"{path.name}: task '{task}' missing. Have: "
                       f"{list(d['samples'].keys())[:6]}...")
    return {int(s["doc_id"]): s for s in d["samples"][task]}


def option_texts(doc: dict) -> list[str]:
    ch = doc.get("choices")
    if isinstance(ch, dict) and "text" in ch:
        return list(ch["text"])
    if isinstance(ch, list) and ch and isinstance(ch[0], str):
        return list(ch)
    mc1 = doc.get("mc1_targets")
    if isinstance(mc1, dict) and "choices" in mc1:
        return list(mc1["choices"])
    raise KeyError(f"no options in doc (keys={list(doc.keys())})")


def get_option_texts(sample: dict) -> list[str]:
    """Option strings for char-norm. ARC path uses doc['choices'] (unchanged);
    GPQA/leaderboard docs lack a standard choices field, so fall back to the
    rendered continuations in `arguments` (option text == the scored continuation)."""
    try:
        return option_texts(sample["doc"])
    except KeyError:
        args = sample.get("arguments")
        if isinstance(args, list) and args and isinstance(args[0], (list, tuple)):
            return [a[1] for a in args]
        raise


def cond_logprobs(sample: dict) -> np.ndarray:
    return np.array([float(r[0]) for r in sample["filtered_resps"]], dtype=np.float64)


def char_lens(texts: list[str]) -> np.ndarray:
    return np.array([max(1, len(t)) for t in texts], dtype=np.float64)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def rank_of(scores: np.ndarray, idx: int) -> int:
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == idx)[0][0])


def z_margin(scores: np.ndarray, i: int, j: int) -> float:
    sd = scores.std()
    if sd < 1e-9:
        return 0.0
    return float((scores[i] - scores[j]) / sd)


def boot_rate_ci(flags, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap mean (rate) + 95% CI for a 0/1 array."""
    x = np.asarray(flags, dtype=np.float64)
    n = len(x)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = np.array([float(x[rng.integers(0, n, n)].mean()) for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("runs/vinf_causal"))
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--null", type=Path, default=None)
    ap.add_argument("--mc", type=Path, default=None)
    ap.add_argument("--arbiter", type=Path, default=None)
    ap.add_argument("--metric", default="acc_norm", choices=["acc_norm", "acc"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    use_norm = args.metric == "acc_norm"

    null_path = args.null or (args.dir / "d_null.json")
    mc_path = args.mc or (args.dir / "d_mc.json")
    print("=" * 80, flush=True)
    print(f"LIFT DIFFICULTY / BURIAL STRATIFICATION | task={args.task} metric={args.metric}", flush=True)
    print("=" * 80, flush=True)

    null = load_task_samples(null_path, args.task)
    mc = load_task_samples(mc_path, args.task)
    ids = sorted(set(null) & set(mc))

    arb = {}
    arb_path = args.arbiter or (args.dir / "arbiter_arc_cot.json")
    have_cot = arb_path.exists()
    if have_cot:
        a = json.load(arb_path.open(encoding="utf-8"))
        for r in a.get("results", []):
            arb[int(r["doc_id"])] = r
        print(f"  arbiter: {arb_path.name}  base_cot_acc_all="
              f"{a.get('summary', {}).get('base_cot_acc_all')}  "
              f"n_unparsed={a.get('summary', {}).get('n_unparsed')}", flush=True)

    # ---- per-item pass ----------------------------------------------------------
    rows = []
    for did in ids:
        s_b, s_m = null[did], mc[did]
        texts = get_option_texts(s_b)
        gold = int(s_b["target"])
        clen = char_lens(texts)
        b_raw, m_raw = cond_logprobs(s_b), cond_logprobs(s_m)
        n_opt = len(texts)
        if not (len(b_raw) == len(m_raw) == n_opt) or gold >= n_opt:
            continue
        b = b_raw / clen if use_norm else b_raw.copy()
        m = m_raw / clen if use_norm else m_raw.copy()
        b_pred, m_pred = int(np.argmax(b)), int(np.argmax(m))
        b_corr, m_corr = int(b_pred == gold), int(m_pred == gold)
        b_rank_gold = rank_of(b, gold)                 # 0=top; base-wrong -> >=1
        b_goldp = float(softmax(b)[gold])
        b_chosen_gold_zm = z_margin(b, b_pred, gold)   # >=0 when base wrong; bigger=more buried
        cot = arb.get(did, {}).get("base_cot_correct")
        rows.append(dict(doc_id=did, gold=gold, n_opt=n_opt,
                         b_corr=b_corr, m_corr=m_corr, fix=int(b_corr == 0 and m_corr == 1),
                         b_rank_gold=b_rank_gold, b_goldp=b_goldp,
                         b_chosen_gold_zm=b_chosen_gold_zm,
                         base_cot_correct=(None if cot is None else int(cot))))

    n = len(rows)
    base_acc = float(np.mean([r["b_corr"] for r in rows]))
    mc_acc = float(np.mean([r["m_corr"] for r in rows]))
    print(f"  aligned items: {n}  | base acc={base_acc:.4f} mc acc={mc_acc:.4f} "
          f"lift={mc_acc - base_acc:+.4f}", flush=True)

    BW = [r for r in rows if r["b_corr"] == 0]
    fix_overall = float(np.mean([r["fix"] for r in BW]))
    print(f"  base-wrong items (BW): {len(BW)}  | overall fix-rate={fix_overall:.3f}", flush=True)

    res = {"task": args.task, "metric": args.metric, "n": n,
           "base_acc": base_acc, "mc_acc": mc_acc, "lift": mc_acc - base_acc,
           "n_base_wrong": len(BW), "fix_rate_overall": fix_overall}

    # ---- Q1a: fix-rate by base gold-rank ----------------------------------------
    print("\n  === Q1a  fix-rate by BASE gold-rank (burial) among base-wrong ===", flush=True)
    print(f"      {'rank':>6s} {'n':>5s} {'fix-rate':>10s}   95% CI", flush=True)
    by_rank = {}
    for rk_label, sel in [("1", lambda r: r["b_rank_gold"] == 1),
                          ("2", lambda r: r["b_rank_gold"] == 2),
                          ("3+", lambda r: r["b_rank_gold"] >= 3)]:
        sub = [r["fix"] for r in BW if sel(r)]
        rate, lo, hi = boot_rate_ci(sub, args.n_boot, args.seed)
        print(f"      {rk_label:>6s} {len(sub):>5d} {rate:>10.3f}   [{lo:.3f},{hi:.3f}]", flush=True)
        by_rank[rk_label] = dict(n=len(sub), fix_rate=rate, ci=[lo, hi])
    res["fix_by_gold_rank"] = by_rank

    # ---- Q1b: fix-rate by chosen-vs-gold z-gap tercile (continuous burial) ------
    print("\n  === Q1b  fix-rate by chosen-vs-gold z-gap tercile (bigger gap = more buried) ===",
          flush=True)
    zg = np.array([r["b_chosen_gold_zm"] for r in BW])
    q33, q67 = np.percentile(zg, [33.3, 66.7])
    by_gap = {}
    print(f"      terciles at z-gap {q33:.3f} / {q67:.3f}", flush=True)
    print(f"      {'tercile':>10s} {'n':>5s} {'fix-rate':>10s}   95% CI", flush=True)
    for lab, sel in [("low(<q33)", lambda v: v <= q33),
                     ("mid", lambda v: q33 < v <= q67),
                     ("high(>q67)", lambda v: v > q67)]:
        sub = [r["fix"] for r in BW if sel(r["b_chosen_gold_zm"])]
        rate, lo, hi = boot_rate_ci(sub, args.n_boot, args.seed)
        print(f"      {lab:>10s} {len(sub):>5d} {rate:>10.3f}   [{lo:.3f},{hi:.3f}]", flush=True)
        by_gap[lab] = dict(n=len(sub), fix_rate=rate, ci=[lo, hi])
    res["fix_by_zgap_tercile"] = by_gap
    # point-biserial corr(fix, z-gap): expect negative if fix falls with burial
    fixarr = np.array([r["fix"] for r in BW], dtype=float)
    r_fix_gap = float(np.corrcoef(fixarr, zg)[0, 1]) if zg.std() > 0 else float("nan")
    r_fix_rank = float(np.corrcoef(fixarr, np.array([r["b_rank_gold"] for r in BW], float))[0, 1])
    print(f"      corr(fix, z-gap)={r_fix_gap:+.3f}   corr(fix, gold-rank)={r_fix_rank:+.3f}  "
          f"(negative = fix falls with burial)", flush=True)
    res["corr_fix_zgap"] = r_fix_gap
    res["corr_fix_rank"] = r_fix_rank

    # ---- Q2: recoverability (CoT-accessibility) -- the de-confounded test -------
    if have_cot:
        cov = [r for r in BW if r["base_cot_correct"] is not None]
        print(f"\n  === Q2  fix x CoT-accessibility among base-wrong (coverage {len(cov)}/{len(BW)}) ===",
              flush=True)
        cot_right = [r for r in cov if r["base_cot_correct"] == 1]   # recoverable
        cot_wrong = [r for r in cov if r["base_cot_correct"] == 0]   # genuinely hard
        for lab, sub in [("CoT-RIGHT (recoverable)", cot_right),
                         ("CoT-WRONG (genuinely hard)", cot_wrong)]:
            rate, lo, hi = boot_rate_ci([r["fix"] for r in sub], args.n_boot, args.seed)
            print(f"      base-wrong & {lab:28s} n={len(sub):>4d}  fix-rate={rate:.3f} "
                  f"[{lo:.3f},{hi:.3f}]", flush=True)
        res["q2_cot"] = {
            "cot_right": dict(n=len(cot_right),
                              fix_rate=boot_rate_ci([r["fix"] for r in cot_right], args.n_boot, args.seed)),
            "cot_wrong": dict(n=len(cot_wrong),
                              fix_rate=boot_rate_ci([r["fix"] for r in cot_wrong], args.n_boot, args.seed)),
        }
        # burial-matched contrast: compare CoT-wrong vs CoT-right at similar z-gap,
        # to peel the gap-closing confound off the recoverability signal.
        if cot_wrong:
            zw = np.array([r["b_chosen_gold_zm"] for r in cot_wrong])
            lo_b, hi_b = float(zw.min()), float(zw.max())
            matched = [r for r in cot_right if lo_b <= r["b_chosen_gold_zm"] <= hi_b]
            mr, mlo, mhi = boot_rate_ci([r["fix"] for r in matched], args.n_boot, args.seed)
            wr, wlo, whi = boot_rate_ci([r["fix"] for r in cot_wrong], args.n_boot, args.seed)
            print(f"      [burial-matched] within z-gap [{lo_b:.2f},{hi_b:.2f}]: "
                  f"CoT-right fix={mr:.3f} (n={len(matched)}) vs CoT-wrong fix={wr:.3f} (n={len(cot_wrong)})",
                  flush=True)
            res["q2_burial_matched"] = dict(zgap_band=[lo_b, hi_b],
                                            cot_right=dict(n=len(matched), fix_rate=mr, ci=[mlo, mhi]),
                                            cot_wrong=dict(n=len(cot_wrong), fix_rate=wr, ci=[wlo, whi]))
        # honest power note
        n_hard = len(cot_wrong)
        res["power_note"] = (f"ARC base CoT acc ~0.97 -> only {n_hard} base-wrong CoT-wrong items; "
                             f"the 'hard' cell is underpowered. Test 'does not replicate on hard' on "
                             f"GPQA cold-MC (options-present), stratified by arbiter, on the pod.")
        print(f"\n  >>> POWER: only {n_hard} genuinely-hard (CoT-wrong) base-wrong items in ARC "
              f"-> 'hard' contrast underpowered HERE. Needs GPQA cold-MC lift (cheap-pod).", flush=True)

    out = args.out or (args.dir / f"lift_difficulty_stratification_{args.task}.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  saved: {out}", flush=True)


if __name__ == "__main__":
    main()
