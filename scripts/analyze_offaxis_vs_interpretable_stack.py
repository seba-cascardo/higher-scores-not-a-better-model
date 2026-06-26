"""Residual #2: does the off-axis adapter's +35 REDUCE to a stack of interpretable
named corrections (length-norm + PMI/answer-prior + contextual calibration), or does
it extract gold-vs-distractor signal those corrections cannot reach?

If an ORACLE-tuned interpretable stack (length exponent alpha, answer-prior weight
beta) on the BASE recovers the lift -> the adapter is "just" a calibrator (claim
deflates: the +35 is reproducible parameter-free by re-scoring). If the best stack
falls far short while the adapter reaches 0.85+ -> the off-axis direction reads
content signal that surface features (functions of the option STRING alone) cannot,
i.e. a learned re-read of the model's own computation, not a surface correction.

Predictors are all FIXED transforms of the cached cold-MC scores (apples-to-apples
with the adapter, which is also a fixed offset):
  score(alpha, beta) = (cond_raw - beta * answer_only_raw) / charlen**alpha
    alpha=1,beta=0 -> acc_norm base (length already out); alpha=1,beta=1 -> PMI-DC;
    answer_only = logP(option | "Answer:") IS the contextual-calibration / answer-prior
    content-free term (Zhao 2102.09690 affine calibration in log-space == subtract it).
  adapter -> mc cond scores (the +35).

The interpretable stack is given the BEST POSSIBLE chance: its (alpha,beta) are
grid-searched to MAXIMIZE ARC acc on the very test set (an oracle upper bound). If
even the oracle stack < adapter, the conclusion holds a fortiori. A held-out split
(tune on half, eval on the other) is reported as a non-oracle sanity.

Also: AUC(gold>distractor) on the BASE-WRONG subset (where the +35 lives) for base /
best-stack / adapter -- the stack's AUC is capped by the residual surface-free signal
(~0.574 from analyze_residual_discrimination); if the adapter clears that cap, the gap
is content signal no surface stack can produce. And R^2 of the adapter's per-option
RE-RANKING (item-centered, removing its global logprob shift) regressed on surface
features -- low R^2 => the adapter is not doing surface correction.

CPU-local, EUR0. Uses the same-scaffold premise file (answer_only_<task>.json).
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


def pair_auc(scores_by_item):
    wins, npairs = 0.0, 0
    for s, gold in scores_by_item:
        g = s[gold]
        for j in range(len(s)):
            if j == gold:
                continue
            npairs += 1
            wins += 1.0 if g > s[j] else (0.5 if g == s[j] else 0.0)
    return (wins / npairs) if npairs else float("nan")


def acc_of(score_fn, items):
    """items: list of (cond_raw, ao_raw, charlen, gold). score_fn(cond,ao,cl)->per-opt."""
    ok = 0
    for cond, ao, cl, gold in items:
        s = score_fn(cond, ao, cl)
        ok += int(int(np.argmax(s)) == gold)
    return ok / len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--premise-file", type=Path,
                    default=Path("runs/vinf_causal/answer_only_arc_challenge.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pf = json.load(args.premise_file.open(encoding="utf-8"))
    task = pf.get("task", "?")
    print("=" * 80, flush=True)
    print(f"RESIDUAL #2: off-axis adapter vs interpretable stack  |  task={task}", flush=True)
    print("=" * 80, flush=True)
    cond_b, cond_m = pf["conditional"]["base"], pf["conditional"]["mc"]
    ao_b, ao_m = pf["answer_only"]["base"], pf["answer_only"]["mc"]
    gold_all, clen_all = pf["gold"], pf["charlen"]
    ids = sorted(set(int(k) for k in cond_b) & set(int(k) for k in cond_m)
                 & set(int(k) for k in ao_b) & set(int(k) for k in gold_all))
    print(f"  items: {len(ids)}  sanity: {pf.get('sanity')}", flush=True)

    # assemble per-item arrays
    base_items, mc_items = [], []   # (cond_raw, ao_raw, charlen, gold)
    for did in ids:
        k = str(did)
        cl = np.array(clen_all[k], dtype=np.float64)
        gold = int(gold_all[k])
        cb = np.array(cond_b[k], dtype=np.float64); ab = np.array(ao_b[k], dtype=np.float64)
        cm = np.array(cond_m[k], dtype=np.float64); am = np.array(ao_m[k], dtype=np.float64)
        if not (len(cb) == len(ab) == len(cm) == len(am) == len(cl)) or gold >= len(cl):
            continue
        base_items.append((cb, ab, cl, gold))
        mc_items.append((cm, am, cl, gold))

    # acc_norm convenience scorers
    accnorm = lambda cond, ao, cl: cond / cl
    pmi = lambda cond, ao, cl: (cond - ao) / cl
    ans_only = lambda cond, ao, cl: ao / cl

    base_accnorm = acc_of(accnorm, base_items)
    base_pmi = acc_of(pmi, base_items)
    base_ans = acc_of(ans_only, base_items)
    adapter_accnorm = acc_of(accnorm, mc_items)
    adapter_pmi = acc_of(pmi, mc_items)
    print(f"\n  --- anchor accuracies (ARC acc_norm) ---", flush=True)
    print(f"      base acc_norm (length out)        {base_accnorm:.4f}", flush=True)
    print(f"      base + PMI/answer-prior           {base_pmi:.4f}", flush=True)
    print(f"      base + answer-only (calib floor)  {base_ans:.4f}", flush=True)
    print(f"      ADAPTER (off-axis) acc_norm       {adapter_accnorm:.4f}   <- the +35", flush=True)
    print(f"      adapter + PMI (over-correct)      {adapter_pmi:.4f}", flush=True)
    lift = adapter_accnorm - base_accnorm

    # ---- oracle interpretable stack: maximize acc over (alpha, beta) on the test set
    alphas = np.round(np.arange(0.0, 2.01, 0.1), 3)
    betas = np.round(np.arange(0.0, 1.51, 0.1), 3)
    best = (-1.0, None, None)
    grid = {}
    for a in alphas:
        for b in betas:
            fn = lambda cond, ao, cl, a=a, b=b: (cond - b * ao) / (cl ** a)
            acc = acc_of(fn, base_items)
            grid[(float(a), float(b))] = acc
            if acc > best[0]:
                best = (acc, float(a), float(b))
    print(f"\n  --- ORACLE interpretable stack (length^alpha + answer-prior*beta, tuned ON test) ---", flush=True)
    print(f"      best stack acc = {best[0]:.4f}  at alpha={best[1]} beta={best[2]}  "
          f"(UPPER BOUND: tuned on the eval itself)", flush=True)
    recov_oracle = (best[0] - base_accnorm) / lift if lift > 1e-9 else float("nan")
    print(f"      lift recovered by oracle stack = {recov_oracle:.1%}  of the +{lift:.3f} "
          f"(adapter = 100%)", flush=True)

    # held-out sanity: tune (alpha,beta) on even items, eval on odd
    even = base_items[0::2]; odd = base_items[1::2]
    best_ho = (-1.0, None, None)
    for a in alphas:
        for b in betas:
            fn = lambda cond, ao, cl, a=a, b=b: (cond - b * ao) / (cl ** a)
            acc = acc_of(fn, even)
            if acc > best_ho[0]:
                best_ho = (acc, float(a), float(b))
    fn_ho = lambda cond, ao, cl: (cond - best_ho[2] * ao) / (cl ** best_ho[1])
    acc_ho = acc_of(fn_ho, odd)
    print(f"      held-out stack acc = {acc_ho:.4f}  (tuned alpha={best_ho[1]} beta={best_ho[2]} "
          f"on even, eval on odd) — non-oracle sanity", flush=True)

    # ---- AUC(gold>distractor) on BASE-WRONG (fixed by base acc_norm argmax) ----
    bw_idx = [i for i, (cond, ao, cl, gold) in enumerate(base_items)
              if int(np.argmax(cond / cl)) != gold]
    print(f"\n  --- AUC(gold>distractor) on BASE-WRONG (n={len(bw_idx)}; where +35 lives) ---", flush=True)
    def auc_on(items, idx, fn):
        return pair_auc([(fn(*it[:3]), it[3]) for it in (items[i] for i in idx)])
    a_base = auc_on(base_items, bw_idx, accnorm)
    a_pmi = auc_on(base_items, bw_idx, pmi)
    fn_best = lambda cond, ao, cl: (cond - best[2] * ao) / (cl ** best[1])
    a_stack = auc_on(base_items, bw_idx, fn_best)
    a_adapt = auc_on(mc_items, bw_idx, accnorm)   # adapter scores on the SAME base-wrong items
    print(f"      base naive (cold-MC)              AUC={a_base:.3f}", flush=True)
    print(f"      base + PMI                        AUC={a_pmi:.3f}", flush=True)
    print(f"      base + oracle stack               AUC={a_stack:.3f}   <- surface CAP", flush=True)
    print(f"      ADAPTER (off-axis)                AUC={a_adapt:.3f}   <- clears the cap?", flush=True)

    # ---- R^2: is the adapter's per-option RE-RANKING explained by surface features? ----
    # item-centered to remove the adapter's global logprob shift; pool options.
    dY, X_len, X_ao = [], [], []
    for (cb, ab, cl, gold), (cm, am, _cl, _g) in zip(base_items, mc_items):
        d = (cm / cl) - (cb / cl)          # adapter change in acc_norm score, per option
        d = d - d.mean()                   # remove global shift (item-centered)
        L = (cl - cl.mean())
        A = (ab / cl); A = A - A.mean()    # answer-prior (surface) per option, centered
        dY.extend(d); X_len.extend(L); X_ao.extend(A)
    dY = np.array(dY); X = np.column_stack([np.ones(len(dY)), np.array(X_len), np.array(X_ao)])
    coef, *_ = np.linalg.lstsq(X, dY, rcond=None)
    pred = X @ coef
    ss_res = float(((dY - pred) ** 2).sum()); ss_tot = float(((dY - dY.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\n  --- adapter re-ranking ~ surface features (item-centered, pooled options) ---", flush=True)
    print(f"      R^2(adapter re-rank | length, answer-prior) = {r2:.3f}  "
          f"(low => NOT a surface correction)", flush=True)

    # ---- verdict ----
    print(f"\n  === VERDICT (residual #2) ===", flush=True)
    if best[0] < base_accnorm + 0.4 * lift and adapter_accnorm > base_accnorm + 0.8 * lift:
        verdict = (f"OFF-AXIS IS NOT AN INTERPRETABLE CALIBRATION: the oracle stack recovers only "
                   f"{recov_oracle:.0%} of the +{lift:.3f} (acc {best[0]:.3f} vs adapter {adapter_accnorm:.3f}), "
                   f"its AUC is capped at {a_stack:.3f} while the adapter reaches {a_adapt:.3f} on base-wrong, "
                   f"and the adapter's re-ranking is {r2:.0%}-explained by surface => the off-axis direction "
                   f"reads content signal (in the activations, not the option string) that length+PMI+calibration "
                   f"cannot. This IS the paper's positive content.")
    else:
        verdict = (f"DEFLATION RISK: the interpretable stack recovers {recov_oracle:.0%} of the lift "
                   f"(acc {best[0]:.3f} vs adapter {adapter_accnorm:.3f}) -> the +35 may be largely a "
                   f"re-scoring/calibration effect; the off-axis framing needs the AUC/R^2 to separate it.")
    print(f"      {verdict}", flush=True)

    payload = dict(task=task, lift=lift,
                   acc=dict(base=base_accnorm, base_pmi=base_pmi, base_ans=base_ans,
                            adapter=adapter_accnorm, adapter_pmi=adapter_pmi,
                            oracle_stack=best[0], oracle_alpha=best[1], oracle_beta=best[2],
                            oracle_recovery=recov_oracle, heldout_stack=acc_ho),
                   auc_base_wrong=dict(n=len(bw_idx), base=a_base, pmi=a_pmi,
                                       oracle_stack=a_stack, adapter=a_adapt),
                   adapter_rerank_r2_surface=r2, verdict=verdict)
    outp = args.out or (args.premise_file.parent / f"offaxis_vs_stack_{task}.json")
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {outp}", flush=True)


if __name__ == "__main__":
    main()
