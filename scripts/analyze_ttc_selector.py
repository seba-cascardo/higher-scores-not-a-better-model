"""P7 (LOCAL, post-pod): TTC k>1 self-consistency vs read-only selector vs adapter.

WHY (audit-remediation P7 / handoff N2, the deployment gate): the draft already admits the
selector "may not win" against a cheap test-time-compute baseline. This turns the admission into
a number. Over the 198 base-wrong ARC items:
  - base self-consistency CoT at k in {3,5} (T=0.7), majority vote  -> accuracy + token cost
  - read-only probe-selector (1 forward + probe)                    -> accuracy + ~0 gen cost
  - adapter cold-MC scoring (1 forward, offset installed)           -> accuracy + ~0 gen cost

GATE (pre-registered in the plan, P7-Step2):
  SC(k=3) acc >= selector acc  -> clean deployment negative: "a cheap reasoning TTC dominates;
      the selector's value is evidentiary, not practical" -> Open-questions closes with a number.
  selector wins in SOME cost regime -> FIRST practical-utility claim -> new section via ledger.

INPUTS (all local): the two pod SC dumps (gen_arbiter_arc_cot_pod.py --subset base_wrong --k {3,5}),
the selector json (probe_as_selector_arc_challenge.json), and arbiter_arc_sets.json (to define
base-wrong and the adapter's fixed set). No GPU.

Usage (local WSL, ~/ml-env):
  python scripts/analyze_ttc_selector.py \
      --sc-k3 runs/ttc/arbiter_arc_sc_k3.json --sc-k5 runs/ttc/arbiter_arc_sc_k5.json \
      --selector runs/vinf_causal/probe_as_selector_arc_challenge.json \
      --sets runs/vinf_causal/arbiter_arc_sets.json \
      --out runs/ttc/ttc_selector_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sc_stats(path, base_wrong):
    """From an arbiter SC dump, restrict to base-wrong doc_ids, return acc + cost."""
    d = json.load(open(path, encoding="utf-8"))
    res = {str(r["doc_id"]): r for r in d["results"]}
    rows = [res[i] for i in base_wrong if i in res]
    n = len(rows)
    correct = sum(int(r["base_cot_correct"]) for r in rows)
    # cost proxy: k * mean generated tokens (all k traces are generated for the vote)
    k = d["summary"].get("k", 1)
    mean_gen = sum(r.get("gen_len", 0) for r in rows) / max(n, 1)
    return {"k": k, "n": n, "acc": round(correct / max(n, 1), 4),
            "n_correct": correct, "mean_gen_tokens": round(mean_gen, 1),
            "cost_gen_tokens_per_item": round(k * mean_gen, 1),
            # From the arbiter summary, whose parser falls back to the last standalone
            # label and therefore undercounts truncated traces. Audited 2026-07-29: the
            # ARC arbiter runs this reads are CLEAN (0 of 594 traces lack an explicit
            # `Answer: X`, 0 reached the 1024 cap), so this number is real HERE -- but it
            # is not a general instrument, hence the name.
            "n_unparsed_ARBITER_FIELD": d["summary"].get("n_unparsed")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-k3", required=True)
    ap.add_argument("--sc-k5", required=True)
    ap.add_argument("--selector", default="runs/vinf_causal/probe_as_selector_arc_challenge.json")
    ap.add_argument("--sets", default="runs/vinf_causal/arbiter_arc_sets.json")
    ap.add_argument("--out", type=Path, default=Path("runs/ttc/ttc_selector_compare.json"))
    args = ap.parse_args()

    sets = json.load(open(args.sets, encoding="utf-8"))
    base_right = {str(x) for x in sets["sets"]["base_right"]}
    base_wrong = [str(it["doc_id"]) for it in sets["items_all"] if str(it["doc_id"]) not in base_right]
    mc_only = {str(x) for x in sets["sets"]["mc_only"]}
    fixed_both = {str(x) for x in sets["sets"]["fixed_by_both"]}
    n_bw = len(base_wrong)

    sc3 = _sc_stats(args.sc_k3, base_wrong)
    sc5 = _sc_stats(args.sc_k5, base_wrong)

    # selector + adapter accuracy over base-wrong (from the flip_matrix / sets; 1 forward cost)
    sel = json.load(open(args.selector, encoding="utf-8"))
    fm = sel.get("flip_matrix", {})
    # base-wrong = fixed + still (the base-incorrect rows); selector "fixed" of them
    sel_fixed = fm.get("fixed"); sel_still = fm.get("still")
    sel_acc = round(sel_fixed / (sel_fixed + sel_still), 4) if sel_fixed is not None else None
    # adapter fixes mc_only + fixed_by_both of the base-wrong set (cold-MC scoring)
    adapter_fixed = len((mc_only | fixed_both) & set(base_wrong))
    adapter_acc = round(adapter_fixed / max(n_bw, 1), 4)

    methods = {
        "base_sc_k3": {**sc3, "cost_type": "generation", "cost_per_item": sc3["cost_gen_tokens_per_item"]},
        "base_sc_k5": {**sc5, "cost_type": "generation", "cost_per_item": sc5["cost_gen_tokens_per_item"]},
        "selector_readonly": {"acc": sel_acc, "n": n_bw, "cost_type": "1_forward+probe",
                              "cost_per_item": 1, "source": "flip_matrix fixed/(fixed+still)"},
        "adapter_coldmc": {"acc": adapter_acc, "n": n_bw, "cost_type": "1_forward",
                           "cost_per_item": 1, "n_fixed": adapter_fixed,
                           "source": "mc_only|fixed_by_both over base_wrong"},
    }

    # GATE
    sc3_acc, sc5_acc = sc3["acc"], sc5["acc"]
    ttc_dominates = (sel_acc is not None) and (sc3_acc >= sel_acc)
    selector_wins_some = (sel_acc is not None) and (sel_acc > max(sc3_acc, sc5_acc))
    if ttc_dominates:
        verdict = ("TTC_DOMINATES: a cheap reasoning TTC (SC k=3) matches/beats the selector -> "
                   "the selector's value is evidentiary, not practical (Open-questions closes).")
    elif selector_wins_some:
        verdict = ("SELECTOR_WINS: read-only selector beats SC at k up to 5 -> a practical-utility "
                   "claim exists (1 forward > k CoT traces at equal-or-better acc) -> new section.")
    else:
        verdict = "MIXED: no clean dominance; report the acc/cost frontier as-is."

    payload = {"n_base_wrong": n_bw, "methods": methods,
               "gate": {"ttc_dominates_at_k3": ttc_dominates,
                        "selector_wins_through_k5": selector_wins_some, "verdict": verdict},
               "read": "acc over the 198 base-wrong ARC items; cost_per_item is k*gen_tokens for SC "
                       "vs 1 forward for selector/adapter. The frontier (acc vs cost) is the result."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["methods"], indent=2))
    print("-" * 60)
    print(f"selector acc(base-wrong)={sel_acc}  adapter acc={adapter_acc}  "
          f"SC k3={sc3_acc} (cost {sc3['cost_gen_tokens_per_item']} tok)  "
          f"SC k5={sc5_acc} (cost {sc5['cost_gen_tokens_per_item']} tok)")
    print(f"VERDICT: {verdict}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
