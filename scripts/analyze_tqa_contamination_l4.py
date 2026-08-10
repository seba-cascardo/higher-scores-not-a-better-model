#!/usr/bin/env python
"""
L4 — TruthfulQA train-on-eval contamination check (SEEN vs UNSEEN held-out).

Closes the TQA HEDGE. The V7-mc contrastive adapter and the matched Align-LoRA were
both trained on TQA *pairs* drawn from the SAME `validation` split they are evaluated
on (TruthfulQA has no native train split). So the +37.25 mc1 lift could be memorisation
of train-on-eval items. This script partitions the eval into:

  SEEN   = items whose question text was in the adapter's TQA training set
           (build_tqa_pairs split='train' = first TRAIN_FRAC of the seed-0 perm), and
  UNSEEN = the held-out remainder the adapter never trained on.

then measures the lift (v7mc - base, and align - base) on EACH subset. If the lift is
memorisation it concentrates on SEEN and collapses on UNSEEN (the Align-LoRA held-out
already collapses 98%->30% per B6b). If the lift is a real answer-prior/selection effect
it survives on UNSEEN.

GATE (handoff 2026-06-29, pre-registered — the split is printed BEFORE any Delta is read):
  TQA HEDGE is REMOVED iff  Delta_UNSEEN(mc1) >= +15pp  AND  Delta(mc2) > +10pp.
  Otherwise TQA stays HEDGE (selection-mc1 only, no truthfulness/generalisation claim).

Matching is by NORMALISED QUESTION TEXT, never doc_id (lm-eval doc_ids do not align with
the build_tqa_pairs seed-0 permutation index — that mismatch is the footgun this avoids).

LOCAL / CPU-only, EUR 0. Reproduces build_tqa_pairs (scripts/train_v7_lofit.py) exactly:
same dataset, same seed-0 torch perm, same validity filter, same TRAIN_FRAC.

Usage (after pulling R5 artefacts from the pod):
  python scripts/analyze_tqa_contamination_l4.py \
      --base   runs/eval_bulletproof/R5_base_0shot \
      --v7mc   runs/eval_bulletproof/R5_v7mc_0shot.json \
      --align  runs/eval_bulletproof/R5_align_0shot \
      --out    runs/eval_bulletproof/tqa_l4_verdict.json
"""
import argparse
import glob
import json
import os
import re
import sys

# --- constants mirrored from scripts/train_v7_lofit.py (build_tqa_pairs) -------------
TRAIN_FRAC = 0.8          # train_v7_lofit.TRAIN_FRAC  (verified 2026-06-29)
SEED = 0                  # seed-0 deterministic perm of the validation split
# The adapter training capped TQA pairs at n; with n >= n_train the SEEN set is the full
# first-80% of the perm. 653 = int(817*0.8) matches corpus_matched.jsonl's 653 TQA rows,
# so the cap did NOT bite — SEEN = all_valid[:n_train]. Override only if the adapter's
# TQA cap was < n_train (then SEEN = all_valid[:cap]).
N_TRAIN_CAP = None        # None = take the whole train split (all_valid[:n_train])

GATE_DELTA_MC1_UNSEEN = 0.15   # >= +15pp on UNSEEN mc1 to clear the HEDGE
GATE_DELTA_MC2 = 0.10          # >  +10pp on mc2 to clear the HEDGE


def normalise_q(q: str) -> str:
    """Whitespace/case-robust but content-exact key. Deliberately NOT aggressive:
    only fold case + collapse internal whitespace + strip surrounding punctuation,
    so distinct questions never collide."""
    q = re.sub(r"\s+", " ", q.strip().lower())
    return q.strip(" \t\r\n.?!\"'")


def build_seen_questions() -> set:
    """Reproduce build_tqa_pairs(split='train', seed=0) EXACTLY and return the set of
    normalised question strings the adapter trained on. Requires `datasets` + `torch`
    (same deps as the trainer)."""
    import torch
    from datasets import load_dataset

    print("[l4] reproducing build_tqa_pairs split=train seed=0 ...", flush=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(ds), generator=rng).tolist()

    all_valid = []  # (question,) in perm order, SAME validity filter as the trainer
    for i in perm:
        ex = ds[i]
        labels = ex["mc1_targets"]["labels"]
        if 1 not in labels:
            continue
        if not any(l == 0 for l in labels):
            continue
        all_valid.append(ex["question"])

    n_total = len(all_valid)
    n_train = int(n_total * TRAIN_FRAC)
    cap = n_train if N_TRAIN_CAP is None else min(N_TRAIN_CAP, n_train)
    seen = {normalise_q(q) for q in all_valid[:cap]}
    print(f"[l4] TQA valid={n_total} | n_train(int {TRAIN_FRAC})={n_train} | "
          f"SEEN questions={len(seen)} (cap={cap})", flush=True)
    return seen


def _iter_samples(path: str, task: str):
    """Yield per-item sample dicts for `task` from either:
      - an lm-eval output DIR (globs **/samples_<task>_*.jsonl), or
      - a v7-runner JSON file with results['samples'][task].
    The per-sample schema is lm-eval's standard sample dict in both cases."""
    if os.path.isdir(path):
        pats = glob.glob(os.path.join(path, "**", f"samples_{task}_*.jsonl"), recursive=True)
        if not pats:
            raise FileNotFoundError(f"no samples_{task}_*.jsonl under {path}")
        with open(sorted(pats)[-1], encoding="utf-8") as f:   # newest if several
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        obj = json.load(open(path, encoding="utf-8"))
        samples = obj.get("samples", {})
        if task not in samples:
            raise KeyError(f"task {task} not in results['samples'] of {path} "
                           f"(have: {list(samples)})")
        yield from samples[task]


def load_per_item(path: str, task: str) -> dict:
    """Return {normalised_question: acc_float} for one arm/task.
    acc is 1/0 for mc1 (argmax-correct) and a probability mass in [0,1] for mc2."""
    out = {}
    for s in _iter_samples(path, task):
        # CONFIRM against a pulled R5 sample file: lm-eval truthful_qa stores the
        # question under doc['question']; acc key is 'acc' or 'acc,none'.
        doc = s.get("doc", {})
        q = doc.get("question")
        if q is None:                                  # CONFIRM: some dumps nest differently
            continue
        acc = s.get("acc", s.get("acc,none"))
        if acc is None:                                # last-resort: derive mc1 from resps
            # CONFIRM: for mc1, argmax of per-choice loglik == target index -> 1 else 0
            raise KeyError(f"no acc field in sample for {task}; inspect keys: {list(s)}")
        out[normalise_q(q)] = float(acc)
    if not out:
        raise RuntimeError(f"loaded 0 items for {task} from {path}")
    return out


def subset_delta(treat: dict, base: dict, keys) -> dict:
    keys = [k for k in keys if k in treat and k in base]
    if not keys:
        return {"n": 0, "treat_acc": None, "base_acc": None, "delta": None}
    t = sum(treat[k] for k in keys) / len(keys)
    b = sum(base[k] for k in keys) / len(keys)
    return {"n": len(keys), "treat_acc": round(t, 4), "base_acc": round(b, 4),
            "delta": round(t - b, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="R5 base eval (lm-eval dir or v7 json)")
    ap.add_argument("--v7mc", required=True, help="R5 v7mc eval (v7-runner json or dir)")
    ap.add_argument("--align", default=None, help="R5 align eval (optional; the "
                    "format-familiarity reference — expected to collapse on UNSEEN)")
    ap.add_argument("--mc1-task", default="truthfulqa_mc1")
    ap.add_argument("--mc2-task", default="truthfulqa_mc2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen_q = build_seen_questions()

    arms = {"base": args.base, "v7mc": args.v7mc}
    if args.align:
        arms["align"] = args.align

    # load mc1 + mc2 per arm
    acc = {}  # acc[arm][task] = {qnorm: acc}
    for arm, path in arms.items():
        acc[arm] = {}
        for tname, task in (("mc1", args.mc1_task), ("mc2", args.mc2_task)):
            acc[arm][tname] = load_per_item(path, task)
            print(f"[l4] {arm}/{tname}: {len(acc[arm][tname])} items from {path}", flush=True)

    # SANITY: print the split membership BEFORE reading any Delta (anti p-hacking).
    eval_qs = list(acc["base"]["mc1"].keys())
    seen_keys = [q for q in eval_qs if q in seen_q]
    unseen_keys = [q for q in eval_qs if q not in seen_q]
    frac_seen = len(seen_keys) / max(len(eval_qs), 1)
    print("=" * 60)
    print(f"[l4] SPLIT SANITY (read this BEFORE the deltas):")
    print(f"     eval items (mc1)= {len(eval_qs)}")
    print(f"     SEEN  = {len(seen_keys)} ({frac_seen:.1%})   "
          f"UNSEEN = {len(unseen_keys)} ({1-frac_seen:.1%})")
    print(f"     expected SEEN ~80% (TRAIN_FRAC={TRAIN_FRAC}); if far off, the "
          f"normalisation or the dataset version drifted — STOP and fix before trusting Delta.")
    if not (0.70 <= frac_seen <= 0.88):
        print(f"     [WARN] SEEN fraction {frac_seen:.1%} outside [70%,88%] — "
              f"matching is suspect; verify normalise_q against the pulled samples.")
    print("=" * 60)

    # deltas per arm / metric / subset
    report = {"split": {"n_eval": len(eval_qs), "n_seen": len(seen_keys),
                        "n_unseen": len(unseen_keys), "frac_seen": round(frac_seen, 4),
                        "train_frac": TRAIN_FRAC, "seed": SEED},
              "deltas": {}}
    for arm in ("v7mc", "align"):
        if arm not in acc:
            continue
        report["deltas"][arm] = {}
        for tname in ("mc1", "mc2"):
            report["deltas"][arm][tname] = {
                "ALL": subset_delta(acc[arm][tname], acc["base"][tname], eval_qs),
                "SEEN": subset_delta(acc[arm][tname], acc["base"][tname], seen_keys),
                "UNSEEN": subset_delta(acc[arm][tname], acc["base"][tname], unseen_keys),
            }

    # GATE — v7mc is the judge of the HEDGE (align is the format-familiarity reference)
    d_unseen_mc1 = report["deltas"]["v7mc"]["mc1"]["UNSEEN"]["delta"]
    d_mc2_all = report["deltas"]["v7mc"]["mc2"]["ALL"]["delta"]
    d_mc2_unseen = report["deltas"]["v7mc"]["mc2"]["UNSEEN"]["delta"]
    pass_mc1 = (d_unseen_mc1 is not None) and (d_unseen_mc1 >= GATE_DELTA_MC1_UNSEEN)
    # use UNSEEN mc2 if available, else ALL — the held-out is the strict reading
    d_mc2_judge = d_mc2_unseen if d_mc2_unseen is not None else d_mc2_all
    pass_mc2 = (d_mc2_judge is not None) and (d_mc2_judge > GATE_DELTA_MC2)
    hedge_removed = bool(pass_mc1 and pass_mc2)

    report["gate"] = {
        "rule": f"Delta_UNSEEN(mc1) >= +{GATE_DELTA_MC1_UNSEEN:.0%} AND Delta(mc2) > +{GATE_DELTA_MC2:.0%}",
        "delta_unseen_mc1": d_unseen_mc1, "pass_mc1": pass_mc1,
        "delta_mc2_judge": d_mc2_judge, "pass_mc2": pass_mc2,
        "verdict": "HEDGE_REMOVED -> TQA bulletproof (held-out)" if hedge_removed
                   else "HEDGE_STAYS -> selection-mc1 only, no truthfulness/generalisation claim",
    }

    json.dump(report, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(report["deltas"], indent=2))
    print("=" * 60)
    print(f"[l4] GATE: {report['gate']['verdict']}")
    print(f"     Delta_UNSEEN(mc1)={d_unseen_mc1} (need >= {GATE_DELTA_MC1_UNSEEN}) -> {pass_mc1}")
    print(f"     Delta(mc2)={d_mc2_judge} (need > {GATE_DELTA_MC2}) -> {pass_mc2}")
    print(f"[l4] wrote {args.out}")
    return 0 if hedge_removed else 0  # exit 0 either way; the verdict is in the json


if __name__ == "__main__":
    sys.exit(main())
