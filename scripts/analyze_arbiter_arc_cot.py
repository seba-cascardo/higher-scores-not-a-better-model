"""ARC arbiter — local analysis (€0). Join base gen-CoT correctness to the offset
sets and report the read.

Run after the pod produces runs/vinf_causal/arbiter_arc_cot.json.

  python scripts/analyze_arbiter_arc_cot.py
"""
import json
import os

SETS = "runs/vinf_causal/arbiter_arc_sets.json"
COT = "runs/vinf_causal/arbiter_arc_cot.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def main():
    if not os.path.exists(COT):
        raise SystemExit(f"{COT} not found — run gen_arbiter_arc_cot_pod.py on the pod first.")
    sets = json.load(open(SETS, encoding="utf-8"))["sets"]
    cot = {r["doc_id"]: r["base_cot_correct"] for r in json.load(open(COT, encoding="utf-8"))["results"]}

    print("ARC arbiter — base gen-CoT correctness by offset set")
    print("=" * 60)
    rows = []
    for name in ["mc_only", "fixed_by_both", "vinf_only", "base_right"]:
        ids = [d for d in sets.get(name, []) if d in cot]
        k = sum(cot[d] for d in ids)
        n = len(ids)
        lo, hi = wilson(k, n)
        acc = k / n if n else float("nan")
        rows.append((name, k, n, acc, lo, hi))
        print(f"  {name:14s} base-CoT-correct {k:3d}/{n:3d} = {acc:.3f}  CI[{lo:.3f},{hi:.3f}]")

    mc = next(r for r in rows if r[0] == "mc_only")
    print("\nREAD (reconcile with W_know = off-axis, ARC fraction 0.064):")
    print(f"  mc_only base-CoT acc = {mc[3]:.3f} (n={mc[2]}).")
    print("  HIGH (~base_right): the fixed items ARE CoT-accessible knowledge. This does NOT")
    print("    make the lift on-axis -- W_know already showed it is off-axis. Synthesis: the")
    print("    adapter uses an OFF-AXIS shortcut to recover answers the model can reason to but")
    print("    cold MC-scoring misranks -> explains D5 (no gen-CoT headroom: base ~ceiling).")
    print("  LOW (base fails even WITH CoT): the fix targets items the model cannot reason out")
    print("    -> the strongest 'pure off-axis artifact' reading.")
    print("  CAVEAT: ARC is near CoT-ceiling -> confirms the benign reading but has LOW power to")
    print("    distinguish 'recovers knowable' from 'fabricates'; a hard task (GPQA) bites harder.")


if __name__ == "__main__":
    main()
