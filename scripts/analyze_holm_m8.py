"""Holm correction over the extended MC family (audit 2026-07-02 §3-B9a).

The pre-registration listed 8 tests; the executed L10 Holm corrected only the m=4
headline lifts. This script recomputes Holm over the McNemar-able MC family:

  m=6 = {ARC, TQA, HSwag, Winogrande}  (headline lifts, b/c counts as published)
      + {MMLU spillover, GPQA-cold spillover}  (recomputed here from local per-item).

gsm8k and lambada are the other two pre-registered tests but are GENERATIVE
(exact-match / perplexity), not paired-MC — there is no McNemar-MC pairing for them,
so the McNemar-Holm family is m=6, not 8. This is documented, not hidden.

Headline McNemar p's are recomputed from their discordant b/c counts (fully reproducible),
so the artifact does not depend on the previously-reported p values.
"""
import argparse
import glob
import json
import os
from pathlib import Path

from scipy.stats import binomtest


# Headline discordant counts, as published; verified against the eval_bulletproof artifacts.
# b = base-right/adapter-wrong ; c = base-wrong/adapter-right.
HEADLINE = {
    "arc_challenge":  {"b": 11, "c": 151},
    "truthfulqa_mc1": {"b": 6,  "c": 155},
    "winogrande":     {"b": 23, "c": 99},
    "hellaswag":      {"b": 39, "c": 107},
}


def mcnemar_p_from_counts(b, c):
    disc = b + c
    return binomtest(min(b, c), disc, 0.5, alternative="two-sided").pvalue if disc > 0 else 1.0


def load_jsonl_acc(path, metric="acc"):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["doc_id"]] = int(round(r[metric]))
    return out


def mmlu_paired_counts(base_dir, v7mc_json, metric="acc"):
    """Pair MMLU per-item base vs v7mc across all 57 subjects -> global b/c."""
    v = json.load(open(v7mc_json, encoding="utf-8"))["samples"]
    b = c = n = 0
    subj_files = sorted(glob.glob(f"{base_dir}/samples_mmlu_*.jsonl"))
    for i, jf in enumerate(subj_files):
        subj = os.path.basename(jf).split("samples_")[1].rsplit("_2026", 1)[0]
        if subj not in v:
            print(f"    [warn] subject {subj} missing in v7mc samples", flush=True)
            continue
        base_acc = load_jsonl_acc(jf, metric)
        v_acc = {r["doc_id"]: int(round(r[metric])) for r in v[subj]}
        ids = set(base_acc) & set(v_acc)
        for did in ids:
            n += 1
            if base_acc[did] == 1 and v_acc[did] == 0:
                b += 1
            elif base_acc[did] == 0 and v_acc[did] == 1:
                c += 1
        if (i + 1) % 15 == 0:
            print(f"    [mmlu {i+1}/{len(subj_files)}] running n={n} b={b} c={c}", flush=True)
    return b, c, n


def gpqa_paired_counts(base_json, mc_json, task="leaderboard_gpqa_diamond", metric="acc_norm"):
    def load(path):
        s = json.load(open(path, encoding="utf-8"))["samples"][task]
        return {r["doc_id"]: int(round(r[metric])) for r in s}
    base, mc = load(base_json), load(mc_json)
    ids = set(base) & set(mc)
    b = sum(1 for i in ids if base[i] == 1 and mc[i] == 0)
    c = sum(1 for i in ids if base[i] == 0 and mc[i] == 1)
    return b, c, len(ids)


def holm(pvals):
    """Holm-Bonferroni. pvals: dict name->p. Returns dict name->(p, p_holm, reject@0.05)."""
    m = len(pvals)
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running_max = {}, 0.0
    for rank, (name, p) in enumerate(order):
        factor = m - rank
        p_holm = min(1.0, p * factor)
        running_max = max(running_max, p_holm)  # enforce monotonicity
        out[name] = {"p_raw": p, "holm_factor": factor, "p_holm": running_max,
                     "reject_0.05": bool(running_max < 0.05)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r10-base-dir",
                    default="runs/eval_bulletproof/R10_base_mmlu/__workspace__models__gemma4-31b-it")
    ap.add_argument("--r10-v7mc", default="runs/eval_bulletproof/R10_v7mc_mmlu.json")
    ap.add_argument("--gpqa-base", default="runs/vinf_causal/d_base_gpqa.json")
    ap.add_argument("--gpqa-mc", default="runs/vinf_causal/d_mc_gpqa.json")
    ap.add_argument("--out", default="runs/eval_bulletproof/holm_m_extended.json")
    args = ap.parse_args()

    pvals, counts = {}, {}
    print("[headline] recomputing 4 McNemar p from the published b/c counts ...", flush=True)
    for task, bc in HEADLINE.items():
        p = mcnemar_p_from_counts(bc["b"], bc["c"])
        pvals[task] = p
        counts[task] = {**bc, "n_discordant": bc["b"] + bc["c"], "sign": "lift(+)"}
        print(f"    {task}: b={bc['b']} c={bc['c']} p={p:.3e}", flush=True)

    print("[mmlu] pairing base vs v7mc per-item (57 subjects) ...", flush=True)
    mb, mc_, mn = mmlu_paired_counts(args.r10_base_dir, args.r10_v7mc)
    pvals["mmlu"] = mcnemar_p_from_counts(mb, mc_)
    counts["mmlu"] = {"b": mb, "c": mc_, "n": mn, "n_discordant": mb + mc_, "sign": "spillover(-)"}
    print(f"    mmlu: b(base+/v7mc-)={mb} c(base-/v7mc+)={mc_} n={mn} p={pvals['mmlu']:.3e}", flush=True)

    print("[gpqa] pairing base vs mc per-item ...", flush=True)
    gb, gc, gn = gpqa_paired_counts(args.gpqa_base, args.gpqa_mc)
    pvals["gpqa_cold"] = mcnemar_p_from_counts(gb, gc)
    counts["gpqa_cold"] = {"b": gb, "c": gc, "n": gn, "n_discordant": gb + gc, "sign": "spillover(-)"}
    print(f"    gpqa_cold: b={gb} c={gc} n={gn} p={pvals['gpqa_cold']:.3e}", flush=True)

    holm6 = holm(pvals)

    # Sensitivity: headline verdicts under m in {4..8}. Largest headline raw p:
    hl_max = max(pvals[t] for t in HEADLINE)
    m8_bound = hl_max * 8
    out = {
        "family_m": len(pvals),
        "family_note": ("McNemar-Holm family = 6 MC tasks; gsm8k and lambada (the other 2 "
                        "pre-registered tests) are generative (exact-match/perplexity), not "
                        "paired-MC, so they carry no McNemar p and are excluded from this family."),
        "counts": counts,
        "pvals_raw": pvals,
        "holm_m6": holm6,
        "headline_max_raw_p": hl_max,
        "sensitivity_m8": {
            "note": ("Holm multiplies the smallest p by m and the k-th smallest by m-k+1; "
                     "the 4 headline lifts are the smallest p's, so even at m=8 the largest "
                     "headline corrected p <= headline_max_raw_p * 8."),
            "headline_corrected_p_upper_bound_at_m8": m8_bound,
            "all_headline_reject_at_m8": bool(m8_bound < 0.05),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n[holm m=6] task: p_raw -> p_holm (reject@0.05)")
    for name, r in sorted(holm6.items(), key=lambda kv: kv[1]["p_raw"]):
        print(f"    {name:16s} {r['p_raw']:.3e} -> {r['p_holm']:.3e}  reject={r['reject_0.05']}")
    print(f"\n[sensitivity] headline corrected-p upper bound at m=8 = {m8_bound:.3e} "
          f"(<0.05: {m8_bound < 0.05})")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
