"""Cold-MC calibration probe: WHY does the base lose answers it knows?

Mechanistic question (ClaraVT, 2026-06-25): the base model KNOWS the answer
(arbiter gen-CoT 123/124, GPQA 0.70) but the cold-MC scoring under chat-template
MISRANKS it. Is the base, on the items the adapter "fixes" (S_flip), (H1) flat /
confused (high entropy, the gold is a near-miss rank-2 it barely loses) or
(H2/H3) confidently WRONG in a spurious direction (low entropy, the gold buried
at rank >=2, mass parked on a wrong option)?

This is the LOCAL, EUR0 readout: it reads the existing lm-eval --log-samples runs
(d_null.json = base, d_mc.json = adapter) which carry per-option SUM-token logprobs
in filtered_resps, plus the arbiter (arbiter_arc_cot.json) that marks S_flip
(in_mc_only) and whether the base solves each item with CoT (base_cot_correct).

We characterise the BASE distribution over options on each subset with
scale-robust statistics (rank of the gold, margins in per-item z-units, normalised
entropy, softmax prob of the gold), CONTRAST S_flip against the control S_base_right,
and measure what the adapter does to S_flip (re-routing: does mc flatten or sharpen?
does it SUPPRESS the base-chosen wrong option or PROMOTE the gold?).

Metric is acc_norm (char-length-normalised logprob), matching the canonical +35.

Pre-registered reads (do not move the goalposts after seeing numbers):
  * H1 (flat/confused)        : S_flip base -> median rank_correct == 1 AND high
                                normalised entropy AND small top1-top2 z-margin.
                                Predicts on-axis sharpening (W_know) would help -> it
                                does not (6.4%), so this is the disfavoured prior.
  * H2/H3 (confidently wrong) : S_flip base -> rank_correct >= 2 OR low entropy with
                                mass on a wrong option; the gold is not a near-miss.
  * re-routing signature      : Delta-entropy(mc-base) < 0 (mc sharpens) and the shift
                                is dominated by SUPPRESSING the base-chosen option
                                (consistent with the theta_mc logit-lens: structural
                                suppression).

Repo conventions: argparse, utf-8 stdout, progress prints with flush, JSON save,
paired bootstrap CIs over items.  CPU-local.
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
# loaders (mirror score_mc_pmi.py)
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


def cond_logprobs(sample: dict) -> np.ndarray:
    return np.array([float(r[0]) for r in sample["filtered_resps"]], dtype=np.float64)


def char_lens(texts: list[str]) -> np.ndarray:
    return np.array([max(1, len(t)) for t in texts], dtype=np.float64)


# ---------------------------------------------------------------------------------
# per-item distribution stats (scale-robust)
# ---------------------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def norm_entropy(scores: np.ndarray) -> float:
    """Entropy of softmax(scores) normalised by log(n) -> [0,1]. 1=flat, 0=peaked.
    Heuristic: softmax over logprob/charlen is NOT a calibrated probability; used as
    a RELATIVE diagnostic across subsets, cross-checked by rank/z-margin stats."""
    p = softmax(scores)
    n = len(p)
    if n <= 1:
        return 0.0
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(n))


def rank_of(scores: np.ndarray, idx: int) -> int:
    """0-based rank of option idx when sorting scores descending (0 = argmax/chosen).
    Ties broken by lower index (np.argsort stable, matches lm-eval argmax convention)."""
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == idx)[0][0])


def z_margin(scores: np.ndarray, i: int, j: int) -> float:
    """(scores[i]-scores[j]) in per-item std units. Scale-invariant."""
    sd = scores.std()
    if sd < 1e-9:
        return 0.0
    return float((scores[i] - scores[j]) / sd)


def boot_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
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
    ap.add_argument("--null", type=Path, default=None, help="base run (default <dir>/d_null.json)")
    ap.add_argument("--mc", type=Path, default=None, help="adapter run (default <dir>/d_mc.json)")
    ap.add_argument("--arbiter", type=Path, default=None,
                    help="arbiter run (optional; cross S_flip with base_cot_correct). "
                         "Defaults to arbiter_arc_cot_v2.json when present, else the pre-fix v1.")
    ap.add_argument("--metric", default="acc_norm", choices=["acc_norm", "acc"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    use_norm = args.metric == "acc_norm"

    null_path = args.null or (args.dir / "d_null.json")
    mc_path = args.mc or (args.dir / "d_mc.json")
    print("=" * 78, flush=True)
    print(f"COLD-MC CALIBRATION PROBE  |  task={args.task}  metric={args.metric}", flush=True)
    print("=" * 78, flush=True)
    print(f"  base : {null_path}", flush=True)
    print(f"  mc   : {mc_path}", flush=True)

    null = load_task_samples(null_path, args.task)
    mc = load_task_samples(mc_path, args.task)
    ids = sorted(set(null) & set(mc))
    print(f"  aligned items: {len(ids)}", flush=True)

    # optional arbiter (ARC only): doc_id -> base_cot_correct, in_mc_only
    # Prefer the fixed-parser v2 run when both are present (same rule as
    # analyze_arbiter_arc_cot.py). The pre-fix v1 left 5 generations unparsed,
    # one of which (doc 128, a numeric option label the letter-parser could not
    # emit) is base-wrong — so reading v1 here silently drops it from
    # S_cot_coldwrong and yields 191 instead of 192.
    arb = {}
    if args.arbiter is not None:
        arb_path = args.arbiter
    else:
        v2 = args.dir / "arbiter_arc_cot_v2.json"
        arb_path = v2 if v2.exists() else (args.dir / "arbiter_arc_cot.json")
    if arb_path.exists():
        a = json.load(arb_path.open(encoding="utf-8"))
        for r in a.get("results", []):
            arb[int(r["doc_id"])] = r
        print(f"  arbiter: {arb_path.name} ({len(arb)} items, "
              f"summary n_mc_only={a.get('summary', {}).get('n_mc_only')})", flush=True)

    # ---- per-item pass: compute base/mc distribution stats -----------------------
    rows = []  # one dict per item
    for did in ids:
        s_b, s_m = null[did], mc[did]
        texts = option_texts(s_b["doc"])
        gold = int(s_b["target"])
        clen = char_lens(texts)
        b_raw = cond_logprobs(s_b)
        m_raw = cond_logprobs(s_m)
        n_opt = len(texts)
        if not (len(b_raw) == len(m_raw) == n_opt) or gold >= n_opt:
            continue
        # length / surface diagnostics (metric-independent: always from raw + charlen)
        shortest = int(np.argmin(clen))
        b_raw_pred = int(np.argmax(b_raw))     # what raw-logprob scoring would pick
        gold_is_shortest = int(gold == shortest)
        rawpred_is_shortest = int(b_raw_pred == shortest)
        rawpred_is_gold = int(b_raw_pred == gold)
        b = b_raw / clen if use_norm else b_raw.copy()
        m = m_raw / clen if use_norm else m_raw.copy()
        b_pred = int(np.argmax(b))
        m_pred = int(np.argmax(m))
        b_corr = int(b_pred == gold)
        m_corr = int(m_pred == gold)

        # base distribution stats
        b_rank_gold = rank_of(b, gold)               # 0=chosen; in S_flip >=1
        b_ent = norm_entropy(b)
        b_goldp = float(softmax(b)[gold])
        # top1-top2 z-margin (how decisive is the base?)
        b_sorted = np.argsort(-b, kind="stable")
        b_top2_zm = z_margin(b, b_sorted[0], b_sorted[1]) if n_opt >= 2 else 0.0
        # confidently-wrong margin: chosen vs gold in z-units (>=0 when base wrong)
        b_chosen_gold_zm = z_margin(b, b_pred, gold)

        # mc distribution stats
        m_rank_gold = rank_of(m, gold)
        m_ent = norm_entropy(m)
        m_goldp = float(softmax(m)[gold])

        # re-routing: softmax mass on the gold and on the BASE-chosen option, both arms
        b_basepred_p = float(softmax(b)[b_pred])
        m_basepred_p = float(softmax(m)[b_pred])     # mass mc leaves on what base chose

        rows.append(dict(
            doc_id=did, n_opt=n_opt, gold=gold,
            b_pred=b_pred, m_pred=m_pred, b_corr=b_corr, m_corr=m_corr,
            b_rank_gold=b_rank_gold, b_ent=b_ent, b_goldp=b_goldp,
            b_top2_zm=b_top2_zm, b_chosen_gold_zm=b_chosen_gold_zm,
            m_rank_gold=m_rank_gold, m_ent=m_ent, m_goldp=m_goldp,
            b_basepred_p=b_basepred_p, m_basepred_p=m_basepred_p,
            gold_is_shortest=gold_is_shortest, rawpred_is_shortest=rawpred_is_shortest,
            rawpred_is_gold=rawpred_is_gold,
            base_cot_correct=arb.get(did, {}).get("base_cot_correct"),
            in_mc_only=arb.get(did, {}).get("in_mc_only"),
        ))

    R = rows
    n = len(R)
    base_acc = np.mean([r["b_corr"] for r in R])
    mc_acc = np.mean([r["m_corr"] for r in R])
    print(f"\n  base acc={base_acc:.4f}  mc acc={mc_acc:.4f}  lift={mc_acc - base_acc:+.4f}  (n={n})",
          flush=True)

    # ---- subsets ------------------------------------------------------------------
    S_flip = [r for r in R if r["b_corr"] == 0 and r["m_corr"] == 1]   # adapter fixes
    S_base_right = [r for r in R if r["b_corr"] == 1]                  # control
    S_break = [r for r in R if r["b_corr"] == 1 and r["m_corr"] == 0]  # adapter breaks
    S_both_wrong = [r for r in R if r["b_corr"] == 0 and r["m_corr"] == 0]
    # ADAPTER-INDEPENDENT subset (de-tautologizes S_flip): base cold-MC WRONG but the base
    # solves it with CoT (arbiter) — defined WITHOUT looking at the adapter. If its base
    # signature matches S_flip, the "flat/buried" readout is not adapter selection.
    S_cot_coldwrong = [r for r in R if r["b_corr"] == 0 and r["base_cot_correct"] == 1]
    print(f"  subsets: S_flip={len(S_flip)}  S_base_right={len(S_base_right)}  "
          f"S_break={len(S_break)}  S_both_wrong={len(S_both_wrong)}  "
          f"S_cot_coldwrong(adapter-indep)={len(S_cot_coldwrong)}", flush=True)

    # cross S_flip with CoT + arbiter's own in_mc_only (consistency check)
    if arb:
        cot = [r["base_cot_correct"] for r in S_flip if r["base_cot_correct"] is not None]
        if cot:
            print(f"  [CoT] S_flip base_cot_correct = {sum(cot)}/{len(cot)} "
                  f"({np.mean(cot):.3f})  <- 'the model KNOWS'", flush=True)
        inmc = [r["in_mc_only"] for r in S_flip if r["in_mc_only"] is not None]
        if inmc:
            print(f"  [consistency] S_flip ∩ arbiter.in_mc_only = {sum(bool(x) for x in inmc)}/"
                  f"{len(inmc)} (should be high if acc_norm flip ~ arbiter mc_only)", flush=True)

    def summarize(name: str, S: list[dict]):
        if not S:
            print(f"\n  --- {name}: EMPTY ---", flush=True)
            return {"n": 0}
        out = {"n": len(S)}
        print(f"\n  --- {name}  (n={len(S)}) ---  [BASE distribution stats]", flush=True)
        for key, label in [
            ("b_rank_gold", "rank of gold (0=chosen)"),
            ("b_ent", "norm entropy [0=peaked,1=flat]"),
            ("b_goldp", "softmax P(gold)"),
            ("b_top2_zm", "top1-top2 z-margin (decisiveness)"),
            ("b_chosen_gold_zm", "chosen-vs-gold z-gap (scale-inv, no softmax)"),
        ]:
            x = np.array([r[key] for r in S], dtype=np.float64)
            mean, lo, hi = boot_mean_ci(x, args.n_boot, args.seed)
            med = float(np.median(x))
            print(f"      {label:38s} mean={mean:+.3f} CI[{lo:+.3f},{hi:+.3f}] median={med:+.3f}",
                  flush=True)
            out[key] = dict(mean=mean, ci=[lo, hi], median=med)
        # rank histogram of the gold (the cleanest H1-vs-H2 discriminator)
        ranks = np.array([r["b_rank_gold"] for r in S])
        maxr = int(ranks.max())
        hist = {int(k): int((ranks == k).sum()) for k in range(maxr + 1)}
        frac_rank1 = float((ranks == 1).mean()) if len(ranks) else float("nan")
        frac_rank_ge2 = float((ranks >= 2).mean()) if len(ranks) else float("nan")
        print(f"      gold-rank hist {hist}   frac(rank==1 near-miss)={frac_rank1:.3f}  "
              f"frac(rank>=2 buried)={frac_rank_ge2:.3f}", flush=True)
        out["gold_rank_hist"] = hist
        out["frac_rank1"] = frac_rank1
        out["frac_rank_ge2"] = frac_rank_ge2
        return out

    res = {
        "S_flip": summarize("S_flip (adapter FIXES — the target items)", S_flip),
        "S_cot_coldwrong": summarize("S_cot_coldwrong (ADAPTER-INDEPENDENT: base cold-wrong but CoT-right)",
                                     S_cot_coldwrong),
        "S_base_right": summarize("S_base_right (control — base already correct)", S_base_right),
        "S_break": summarize("S_break (adapter BREAKS)", S_break),
        "S_both_wrong": summarize("S_both_wrong", S_both_wrong),
    }

    # ---- length / surface-form diagnostics (reconciles raw-vs-norm) ---------------
    # The raw-vs-acc_norm flip is the tell: is the gross bias the LENGTH of the option
    # (raw-logprob favours the shortest continuation)? If in S_flip the shortest option
    # is usually NOT the gold and raw-scoring picks the shortest, then char-norm is what
    # partially rescues the gold, and the residual cold-MC failure is what's left.
    print(f"\n  === LENGTH / SURFACE-FORM (metric-independent: raw logprob + charlen) ===", flush=True)
    print(f"      {'subset':16s} {'n':>4s}  P(gold=shortest)  P(rawpick=shortest)  P(rawpick=gold)", flush=True)
    surf = {}
    for name, S in [("S_flip", S_flip), ("S_cot_coldwrong", S_cot_coldwrong),
                    ("S_base_right", S_base_right),
                    ("S_break", S_break), ("S_both_wrong", S_both_wrong)]:
        if not S:
            continue
        gsh = float(np.mean([r["gold_is_shortest"] for r in S]))
        psh = float(np.mean([r["rawpred_is_shortest"] for r in S]))
        pg = float(np.mean([r["rawpred_is_gold"] for r in S]))
        print(f"      {name:16s} {len(S):>4d}      {gsh:.3f}            {psh:.3f}             {pg:.3f}",
              flush=True)
        surf[name] = dict(gold_is_shortest=gsh, rawpick_is_shortest=psh, rawpick_is_gold=pg)
    res["length_surface"] = surf

    # ---- what does the adapter DO to S_flip? (re-routing signature) ---------------
    if S_flip:
        print(f"\n  === RE-ROUTING: what mc does to S_flip (n={len(S_flip)}) ===", flush=True)
        d_ent = np.array([r["m_ent"] - r["b_ent"] for r in S_flip])
        d_goldp = np.array([r["m_goldp"] - r["b_goldp"] for r in S_flip])
        d_rank = np.array([r["m_rank_gold"] - r["b_rank_gold"] for r in S_flip], dtype=float)
        # mass mc leaves on the option the BASE wrongly chose (low = mc suppressed it)
        d_basepred_p = np.array([r["m_basepred_p"] - r["b_basepred_p"] for r in S_flip])
        b_basepred_p = np.array([r["b_basepred_p"] for r in S_flip])
        m_basepred_p = np.array([r["m_basepred_p"] for r in S_flip])
        for label, x in [
            ("Δ norm-entropy (mc-base)   <0 = mc SHARPENS", d_ent),
            ("Δ softmax P(gold) (mc-base) >0 = mc promotes gold", d_goldp),
            ("Δ gold rank (mc-base)      <0 = mc lifts gold up", d_rank),
            ("base P(base-chosen-wrong)  (mass to remove)", b_basepred_p),
            ("mc   P(base-chosen-wrong)  (mass left)", m_basepred_p),
            ("Δ P(base-chosen-wrong)     <0 = mc SUPPRESSES it", d_basepred_p),
        ]:
            mean, lo, hi = boot_mean_ci(x, args.n_boot, args.seed)
            print(f"      {label:42s} mean={mean:+.3f} CI[{lo:+.3f},{hi:+.3f}]", flush=True)
        # decompose the move: is the re-rank driven more by suppressing the wrong
        # option or promoting the gold? compare |Δ mass on base-chosen| vs Δ mass on gold
        suppress = float(np.mean(b_basepred_p - m_basepred_p))   # how much wrong-mass removed
        promote = float(np.mean(d_goldp))                        # how much gold-mass added
        print(f"      decomposition: mean wrong-mass REMOVED={suppress:+.3f}  vs  "
              f"gold-mass ADDED={promote:+.3f}  -> "
              f"{'SUPPRESSION-dominated' if suppress > promote else 'PROMOTION-dominated'}",
              flush=True)
        res["re_routing"] = dict(
            d_entropy=boot_mean_ci(d_ent, args.n_boot, args.seed),
            d_goldp=boot_mean_ci(d_goldp, args.n_boot, args.seed),
            d_rank=boot_mean_ci(d_rank, args.n_boot, args.seed),
            wrong_mass_removed=suppress, gold_mass_added=promote,
            mechanism="suppression" if suppress > promote else "promotion",
        )

    # ---- pre-registered read ------------------------------------------------------
    if S_flip and S_base_right:
        fr = res["S_flip"]
        print(f"\n  === PRE-REGISTERED READ (S_flip base) ===", flush=True)
        med_rank = fr["b_rank_gold"]["median"]
        ent = fr["b_ent"]["mean"]
        f1, fge2 = fr["frac_rank1"], fr["frac_rank_ge2"]
        ent_ctrl = res["S_base_right"]["b_ent"]["mean"]
        goldp = fr["b_goldp"]["mean"]
        goldp_ctrl = res["S_base_right"]["b_goldp"]["mean"]
        print(f"      median gold-rank={med_rank:.1f}  | frac near-miss(rank1)={f1:.2f}  "
              f"frac buried(rank>=2)={fge2:.2f}", flush=True)
        print(f"      norm-entropy S_flip={ent:.3f} vs control S_base_right={ent_ctrl:.3f}", flush=True)
        print(f"      P(gold) S_flip={goldp:.3f} vs control={goldp_ctrl:.3f}", flush=True)
        if fge2 >= 0.5 or (goldp < 0.5 * goldp_ctrl):
            read = ("H2/H3 CONFIDENTLY-WRONG: on the items it knows, the base buries the gold "
                    "(rank>=2 majority and/or P(gold) far below control) -> spurious confidence, "
                    "not a flat near-miss. Consistent with W_know failing (no on-axis headroom).")
        elif f1 >= 0.6 and ent >= 0.85:
            read = ("H1 FLAT/NEAR-MISS: gold is mostly rank-2 with high entropy -> base barely "
                    "loses, decision is soft. (Would predict on-axis sharpening helps — tension "
                    "with W_know 6.4%; re-examine.)")
        else:
            read = "MIXED: see the rank histogram and the S_flip-vs-control contrast above."
        print(f"      -> {read}", flush=True)
        res["read"] = read

    payload = dict(task=args.task, metric=args.metric, n=n,
                   base_acc=float(base_acc), mc_acc=float(mc_acc),
                   lift=float(mc_acc - base_acc), subsets=res)
    out = args.out or (args.dir / f"cold_mc_calibration_{args.task}.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {out}", flush=True)


if __name__ == "__main__":
    main()
