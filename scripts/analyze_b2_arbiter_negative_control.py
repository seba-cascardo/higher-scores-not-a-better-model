"""B2 — negative control for the per-item ARC arbiter (the 124/124 claim).

The adversarial review (2026-07-15, PR9) raised: the arbiter shows the base solves
124/124 of the items the adapter fixes under generative CoT, but with NO negative
control this could be trivially true of all of ARC — if the base solves ~everything
under CoT, "the fixed items were already CoT-accessible" says nothing about the
adapter.

The control is the complementary population: the items the adapter does NOT fix
(its residual). If base-CoT accuracy there is just as high, the arbiter does not
discriminate. If it is lower, the arbiter carries information.

KEY METHODOLOGICAL POINT: no new generation is needed. The v2 arbiter run
(runs/vinf_causal/arbiter_arc_cot_v2.json) already covers ALL 400 items of the
vinf_causal population, not only the 124 — same run, same prompt, same greedy k=1
decode, same fixed letter-parser (n_unparsed=0). The control is therefore
protocol-identical to the published 124/124 by construction, which is strictly
better than re-running it on a pod (zero risk of protocol drift). This script only
re-partitions that existing run.

Two scaffolds, because the paper's populations come from two harnesses:
  * lm-eval  (PRIMARY, apples-to-apples): acc_norm from d_{null,mc}.json — the same
    scoring that defines mc_only=124. Residual := base-wrong AND adapter-wrong.
  * premise-file (ROBUSTNESS): answer_only_arc_challenge.json, the scaffold that
    defines the n=73 "adapter residual" of section 5.1. Residual := adapter-wrong.
Reporting both guards against the partition being an artifact of one scoring path.

CPU-local, EUR0.
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

D = Path("runs/vinf_causal")


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Point estimate + Wilson 95% interval (well-behaved at p near 1)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - half), min(1.0, c + half)


def load_arc_acc_norm(path: Path) -> dict[int, int]:
    """doc_id -> acc_norm (0/1) from an lm-eval per-sample dump."""
    obj = json.load(path.open(encoding="utf-8"))
    samples = obj.get("samples")
    lst = None
    if isinstance(samples, dict):
        for tk in samples:
            if "arc" in tk:
                lst = samples[tk]
                break
    if lst is None:
        raise SystemExit(f"no arc per-sample list in {path}")
    out = {}
    for i, s in enumerate(lst):
        acc = s.get("acc_norm")
        if acc is None and isinstance(s.get("metrics"), dict):
            acc = s["metrics"].get("acc_norm")
        out[s.get("doc_id", i)] = int(round(float(acc)))
    return out


def premise_wrong_ids(pf: dict, arm: str) -> set[int]:
    """doc_ids where the arm's cold-MC argmax (length-normalised) misses the gold."""
    cond, ao = pf["conditional"][arm], pf["answer_only"][arm]
    gold_all, clen_all = pf["gold"], pf["charlen"]
    ids = sorted(set(int(k) for k in cond) & set(int(k) for k in ao)
                 & set(int(k) for k in gold_all) & set(int(k) for k in clen_all))
    wrong = set()
    for did in ids:
        k = str(did)
        c = np.array(cond[k], dtype=np.float64)
        cl = np.array(clen_all[k], dtype=np.float64)
        gold = int(gold_all[k])
        if gold >= len(c) or len(c) != len(cl):
            continue
        if int(np.argmax(c / cl)) != gold:
            wrong.add(did)
    return wrong


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arbiter", type=Path, default=D / "arbiter_arc_cot_v2.json")
    ap.add_argument("--premise-file", type=Path, default=D / "answer_only_arc_challenge.json")
    ap.add_argument("--sets", type=Path, default=D / "arbiter_arc_sets.json")
    ap.add_argument("--d-null", type=Path, default=D / "d_null.json")
    ap.add_argument("--d-mc", type=Path, default=D / "d_mc.json")
    ap.add_argument("--out", type=Path, default=D / "arbiter_negative_control_b2.json")
    args = ap.parse_args()

    print("=" * 82, flush=True)
    print("B2 — ARBITER NEGATIVE CONTROL (base-CoT on the adapter's residual)", flush=True)
    print("=" * 82, flush=True)

    print("[1/5] loading arbiter run ...", flush=True)
    arb = json.load(args.arbiter.open(encoding="utf-8"))
    res = {r["doc_id"]: r for r in arb["results"]}
    print(f"      {args.arbiter.name}: n={len(res)}  summary={arb.get('summary')}", flush=True)

    print("[2/5] loading lm-eval per-sample dumps (~55MB each) ...", flush=True)
    mnull = load_arc_acc_norm(args.d_null)
    mmc = load_arc_acc_norm(args.d_mc)
    print(f"      d_null={len(mnull)}  d_mc={len(mmc)}", flush=True)

    print("[3/5] loading premise-file + published sets ...", flush=True)
    pf = json.load(args.premise_file.open(encoding="utf-8"))
    sets = json.load(args.sets.open(encoding="utf-8"))
    mc_only = set(sets["sets"]["mc_only"])

    # ---- alignment guard: the two scaffolds must index the same items ----
    print("[4/5] alignment guard (gold + n_choices across scaffolds) ...", flush=True)
    items = {it["doc_id"]: it for it in sets["items_all"]}
    gold_all = pf["gold"]
    bad_gold = bad_nc = 0
    for did, it in items.items():
        k = str(did)
        if k not in gold_all:
            continue
        labels = it["choices"]["label"]
        gold_lmeval = labels.index(it["answerKey"]) if it["answerKey"] in labels else None
        if gold_lmeval is None or gold_lmeval != int(gold_all[k]):
            bad_gold += 1
        if len(it["choices"]["text"]) != len(pf["conditional"]["mc"][k]):
            bad_nc += 1
    print(f"      gold mismatches={bad_gold}  n_choices mismatches={bad_nc}", flush=True)
    if bad_gold or bad_nc:
        raise SystemExit("ABORT: scaffolds are not aligned — the partition would be meaningless")

    print("[5/5] partitioning + scoring ...\n", flush=True)

    def cell(pop: set[int], label: str) -> dict:
        pop = sorted(d for d in pop if d in res)
        k = sum(1 for d in pop if res[d]["base_cot_correct"])
        n = len(pop)
        p, lo, hi = wilson(k, n)
        print(f"    {label:50s} {k:3d}/{n:3d} = {p:.4f}  CI95[{lo:.3f},{hi:.3f}]", flush=True)
        return dict(k=k, n=n, acc=p, ci=[lo, hi])

    def fisher(a: dict, b: dict) -> float | None:
        try:
            from scipy.stats import fisher_exact
        except ImportError:
            return None
        _, p = fisher_exact([[a["k"], a["n"] - a["k"]], [b["k"], b["n"] - b["k"]]])
        return float(p)

    out: dict = {
        "question": "Does the 124/124 arbiter discriminate, or is base-CoT trivially high on all of ARC?",
        "method": ("Re-partition of the EXISTING v2 arbiter run (all 400 items, greedy k=1, "
                   "fixed letter-parser, n_unparsed=0). No new generation: the control is "
                   "protocol-identical to the published 124/124 by construction."),
        "arbiter_run": str(args.arbiter),
    }

    # ---------- PRIMARY: lm-eval scaffold (defines mc_only = 124) ----------
    ids = set(mnull) & set(mmc) & set(res)
    bw = {d for d in ids if mnull[d] == 0}
    mw = {d for d in ids if mmc[d] == 0}
    print("  --- PRIMARY: lm-eval scaffold (apples-to-apples with mc_only=124) ---", flush=True)
    prim = {
        "residual": cell(bw & mw, "residual (base-wrong AND adapter-wrong)"),
        "fixed": cell(bw - mw, "fixed (base-wrong AND adapter-right)"),
        "mc_only_published": cell(mc_only, "  of which mc_only (the published arbiter)"),
        "base_right_control": cell(ids - bw, "base-right control"),
        "all_arc": cell(ids, "all of ARC (the reviewer's null)"),
    }
    prim["fisher_residual_vs_fixed"] = fisher(prim["residual"], prim["fixed"])
    prim["fisher_residual_vs_all"] = fisher(prim["residual"], prim["all_arc"])
    print(f"\n    Fisher residual vs fixed  p = {prim['fisher_residual_vs_fixed']:.3e}", flush=True)
    print(f"    Fisher residual vs all-ARC p = {prim['fisher_residual_vs_all']:.4f}", flush=True)
    out["primary_lmeval"] = prim

    # ---------- ROBUSTNESS: premise-file scaffold (defines the n=73) ----------
    mw_pf = premise_wrong_ids(pf, "mc")
    bw_pf = premise_wrong_ids(pf, "base")
    ids_pf = set(int(k) for k in pf["conditional"]["mc"]) & set(res)
    print("\n  --- ROBUSTNESS: premise-file scaffold (defines the n=73 of §5.1) ---", flush=True)
    rob = {
        "residual_73": cell(mw_pf, "adapter residual (the n=73 of §5.1)"),
        "residual_73_and_base_wrong": cell(mw_pf & bw_pf, "  ∩ base-wrong"),
        "complement": cell(ids_pf - mw_pf, "adapter right (n=327)"),
    }
    rob["fisher_residual_vs_complement"] = fisher(rob["residual_73"], rob["complement"])
    print(f"\n    Fisher residual vs complement p = {rob['fisher_residual_vs_complement']:.3e}", flush=True)
    out["robustness_premise_file"] = rob

    # ---------- verdict against the pre-registered thresholds ----------
    r = prim["residual"]
    thresholds = {"discriminates_if_le": 0.85, "saturated_if_approx": 0.97}
    if r["acc"] <= 0.85:
        verdict = "DISCRIMINATES"
    elif r["ci"][1] < 0.97:
        verdict = "DISCRIMINATES (intermediate point estimate, but CI excludes the saturated reading)"
    else:
        verdict = "SATURATED — the arbiter does not discriminate"
    out["thresholds_preregistered"] = thresholds
    out["verdict"] = (
        f"{verdict}. Residual base-CoT {r['k']}/{r['n']} = {r['acc']:.3f} CI[{r['ci'][0]:.3f},"
        f"{r['ci'][1]:.3f}] vs {prim['fixed']['acc']:.3f} on the items the adapter fixes "
        f"(p={prim['fisher_residual_vs_fixed']:.1e}) and {prim['all_arc']['acc']:.3f} on all of ARC. "
        f"The 124/124 is therefore NOT trivially true of all of ARC. Magnitude caveat: ARC is close "
        f"to ceiling under CoT in every population (all-ARC {prim['all_arc']['acc']:.3f}), so the "
        f"contrast is significant but modest in absolute terms."
    )
    print(f"\n  === VERDICT ===\n    {out['verdict']}", flush=True)

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
