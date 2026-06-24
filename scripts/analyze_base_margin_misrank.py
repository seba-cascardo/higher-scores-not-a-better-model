#!/usr/bin/env python3
"""PC-A5 — base-margin misrank proxy (ARC option-ranking geometry, by arbiter stratum).

Question (formalized from a validated inline analysis):
    On the ``mc_only`` stratum, does the BASE model put gold near the top (a near-tie
    that a gentle nudge would flip), or does it put gold *far down* the option ranking
    so that the V7-mc offset has to perform a large override to win?

This script characterizes, per ARC item and per arbiter stratum (mc_only /
fixed_by_both / base_right), the option-ranking geometry of the BASE scorer
(``f_null.json``) and how the V7-mc OFFSET scorer (``f_mc.json``) re-ranks it:

  * winner-margin  = top1_logprob - top2_logprob  (how decisive the winner is)
  * gold rank      = rank of the gold option (0 = gold wins; >=2 means "rank>=3")
  * gold-gap-to-top = top1_logprob - gold_logprob (>=0; 0 iff gold wins)

The headline finding to reproduce (mc_only, acc_norm view — the primary view, defined
above and printed by main()):
  * base puts gold at rank>=3 ~64% of the time  (NOT a near-tie)
  * the offset moves gold to winner (rank0) ~99%  (a big override, not a tiebreak)
  (Provenance note: the raw-acc view gives ~56% / ~94% for these two numbers; the older
  56%/94% figures were the raw-acc view, not acc_norm.)
This supports PC-A6 (over-scaled functional separator) and complicates the reading
of F as a *gentle* de-collapse of a near-tie.

CRITICAL vs the inline draft: we compute BOTH a raw (acc) view and a
length-normalized (acc_norm) view. lm-eval ``arc_challenge`` reports ``acc_norm``,
which divides each option's summed logprob by the length of the choice continuation
text (``doc_to_choice`` = ``choices.text``). The mc_only stratum is defined by
acc_norm, so the acc_norm view is the primary one; raw acc is reported alongside as a
sanity check. We normalize by character length (lm-eval's unit); byte length is
equivalent for ARC choices (validated: both reproduce lm-eval acc_norm 400/400).

Inputs (LOCAL):
  runs/wknow_causal/f_null.json   base scorer  (lm-eval result JSON)
  runs/wknow_causal/f_mc.json     V7-mc offset scorer (lm-eval result JSON)
  runs/vinf_causal/arbiter_arc_sets.json   sets.{mc_only,fixed_by_both,base_right}

Output:
  runs/wknow_causal/base_margin_misrank.json

Pure json + numpy (no torch). Runs locally.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_NULL = os.path.join(REPO, "runs", "wknow_causal", "f_null.json")
F_MC = os.path.join(REPO, "runs", "wknow_causal", "f_mc.json")
ARBITER = os.path.join(REPO, "runs", "vinf_causal", "arbiter_arc_sets.json")
OUT = os.path.join(REPO, "runs", "wknow_causal", "base_margin_misrank.json")

TASK = "arc_challenge"
STRATA = ["mc_only", "fixed_by_both", "base_right"]


def log(msg):
    print(msg, flush=True)


def load_samples(path):
    """Return {doc_id: {'lp': [logprob per option], 'lens': [char len per option],
    'target': gold_idx}} for the arc_challenge task."""
    log(f"  loading {os.path.relpath(path, REPO)} ...")
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    samples = d["samples"][TASK]
    out = {}
    for ex in samples:
        did = int(ex["doc_id"])
        lp = [float(r[0]) for r in ex["filtered_resps"]]
        texts = ex["doc"]["choices"]["text"]
        # char length of the choice continuation (lm-eval acc_norm unit).
        lens = [max(1, len(t)) for t in texts]
        out[did] = {
            "lp": np.asarray(lp, dtype=np.float64),
            "lens": np.asarray(lens, dtype=np.float64),
            "target": int(ex["target"]),
        }
    log(f"    -> {len(out)} {TASK} docs")
    return out


def rank_geometry(scores, target):
    """Given per-option scores (higher=better) and gold idx, return ranking stats.

    rank0 = gold wins. winner_margin = top1 - top2. gold_gap = top1 - gold_score.
    """
    order = np.argsort(-scores, kind="stable")  # best first
    rank_of_gold = int(np.where(order == target)[0][0])
    top1 = float(scores[order[0]])
    top2 = float(scores[order[1]]) if len(order) > 1 else float("-inf")
    gold_score = float(scores[target])
    return {
        "gold_rank": rank_of_gold,
        "winner_margin": top1 - top2,
        "gold_gap_to_top": top1 - gold_score,  # >=0, 0 iff gold wins
        "gold_wins": rank_of_gold == 0,
    }


def summarize(values):
    """Robust summary of a 1-D float array."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def rank_hist(ranks, n_opts_max=8):
    """Distribution over gold rank, plus the headline rank>=2 ('rank>=3') fraction."""
    ranks = np.asarray(ranks, dtype=int)
    n = ranks.size
    hist = {}
    for r in range(n_opts_max):
        c = int(np.sum(ranks == r))
        if c:
            hist[f"rank{r}"] = c
    frac_rank0 = float(np.mean(ranks == 0)) if n else 0.0
    # "rank>=3" in 1-based human terms == 0-based index >= 2.
    frac_rank_ge3 = float(np.mean(ranks >= 2)) if n else 0.0
    return {
        "n": int(n),
        "counts": hist,
        "frac_rank0_gold_wins": frac_rank0,
        "frac_rank_ge3_gold_3rd_or_worse": frac_rank_ge3,
    }


def view_for_stratum(base, off, doc_ids, normalize):
    """Compute geometry for one stratum under one scoring view (raw or norm).

    normalize=False -> raw summed logprob (acc view).
    normalize=True  -> logprob / char_len (acc_norm view).
    """
    base_ranks, off_ranks = [], []
    base_margin, off_margin = [], []
    base_gap, off_gap = [], []
    n_offset_override = 0   # gold not rank0 at base but rank0 after offset
    n_base_gold_wins = 0
    n_off_gold_wins = 0
    missing = 0

    for did in doc_ids:
        b = base.get(did)
        o = off.get(did)
        if b is None or o is None:
            missing += 1
            continue
        if normalize:
            bs = b["lp"] / b["lens"]
            os_ = o["lp"] / o["lens"]
        else:
            bs = b["lp"]
            os_ = o["lp"]
        tgt = b["target"]
        assert tgt == o["target"], f"target mismatch doc {did}"

        gb = rank_geometry(bs, tgt)
        go = rank_geometry(os_, tgt)

        base_ranks.append(gb["gold_rank"])
        off_ranks.append(go["gold_rank"])
        base_margin.append(gb["winner_margin"])
        off_margin.append(go["winner_margin"])
        base_gap.append(gb["gold_gap_to_top"])
        off_gap.append(go["gold_gap_to_top"])

        if gb["gold_wins"]:
            n_base_gold_wins += 1
        if go["gold_wins"]:
            n_off_gold_wins += 1
        if (not gb["gold_wins"]) and go["gold_wins"]:
            n_offset_override += 1

    n = len(base_ranks)
    return {
        "n": n,
        "missing_docs": missing,
        "base": {
            "gold_rank_hist": rank_hist(base_ranks),
            "winner_margin": summarize(base_margin),
            "gold_gap_to_top": summarize(base_gap),
            "gold_win_rate": (n_base_gold_wins / n) if n else 0.0,
        },
        "offset": {
            "gold_rank_hist": rank_hist(off_ranks),
            "winner_margin": summarize(off_margin),
            "gold_gap_to_top": summarize(off_gap),
            "gold_win_rate": (n_off_gold_wins / n) if n else 0.0,
        },
        "override": {
            # of items where base did NOT have gold winning, what fraction does the
            # offset flip to gold-winner? (the "big override vs tiebreak" question)
            "n_base_gold_not_winner": n - n_base_gold_wins,
            "n_offset_flips_to_gold_winner": n_offset_override,
            "frac_offset_moves_gold_to_winner_overall": (n_off_gold_wins / n) if n else 0.0,
            "frac_offset_rescues_of_base_misses": (
                n_offset_override / (n - n_base_gold_wins) if (n - n_base_gold_wins) else 0.0
            ),
        },
    }


def main():
    log("[PC-A5] base-margin misrank proxy — ARC option-ranking geometry by stratum")
    log("[1/4] loading inputs ...")
    base = load_samples(F_NULL)
    off = load_samples(F_MC)
    log(f"  loading {os.path.relpath(ARBITER, REPO)} ...")
    with open(ARBITER, "r", encoding="utf-8") as fh:
        arb = json.load(fh)
    sets = arb["sets"]
    for k in STRATA:
        log(f"    stratum {k}: n={len(sets[k])}")

    log("[2/4] computing geometry per stratum (raw=acc and norm=acc_norm) ...")
    result = {
        "purpose": "PC-A5 base-margin misrank proxy: ARC option-ranking geometry "
        "(base vs V7-mc offset) by arbiter stratum, raw (acc) and norm (acc_norm).",
        "inputs": {
            "base": os.path.relpath(F_NULL, REPO),
            "offset": os.path.relpath(F_MC, REPO),
            "arbiter_sets": os.path.relpath(ARBITER, REPO),
            "task": TASK,
        },
        "normalization_note": (
            "norm view divides each option summed-logprob by char length of the "
            "choice continuation (lm-eval arc_challenge acc_norm unit); byte length "
            "is equivalent for these ASCII-dominant choices (winner-identical)."
        ),
        "strata": {},
    }

    total = len(STRATA) * 2
    step = 0
    for stratum in STRATA:
        doc_ids = [int(x) for x in sets[stratum]]
        result["strata"][stratum] = {}
        for view_name, norm in [("raw_acc", False), ("norm_acc_norm", True)]:
            step += 1
            log(f"  [{step}/{total}] stratum={stratum} view={view_name} "
                f"(n={len(doc_ids)}) ...")
            result["strata"][stratum][view_name] = view_for_stratum(
                base, off, doc_ids, normalize=norm
            )

    log("[3/4] writing output ...")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log(f"  wrote {os.path.relpath(OUT, REPO)}")

    log("[4/4] HEADLINE (mc_only, acc_norm view) ===========================")
    mc = result["strata"]["mc_only"]["norm_acc_norm"]
    bh = mc["base"]["gold_rank_hist"]
    oh = mc["offset"]["gold_rank_hist"]
    log(f"  n={mc['n']}")
    log(f"  BASE gold-wins rate (rank0)        = {bh['frac_rank0_gold_wins']:.3f}")
    log(f"  BASE gold rank>=3 (3rd-or-worse)   = {bh['frac_rank_ge3_gold_3rd_or_worse']:.3f}")
    log(f"  BASE winner-margin median          = {mc['base']['winner_margin']['median']:.3f}")
    log(f"  BASE gold-gap-to-top median        = {mc['base']['gold_gap_to_top']['median']:.3f}")
    log(f"  OFFSET gold-wins rate (rank0)      = {oh['frac_rank0_gold_wins']:.3f}")
    log(f"  OFFSET moves gold->winner overall  = {mc['override']['frac_offset_moves_gold_to_winner_overall']:.3f}")
    log(f"  OFFSET rescues of base-misses      = {mc['override']['frac_offset_rescues_of_base_misses']:.3f}")
    log("  ================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
