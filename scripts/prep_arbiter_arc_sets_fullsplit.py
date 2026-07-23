"""Build the FULL-SPLIT ARC arbiter sets (audit 2026-07-23 C-3; closes R-2 "shared sample").

Same construction as prep_arbiter_arc_sets.py but over the FULL ARC-Challenge test split
(~1172 items) and with only TWO legs (base + mc adapter — no v_inf leg): the C-3 question
is whether the canonical 400-slice arbiter verdict (base-CoT accessibility of the
adapter's cold-MC fixes) holds on the whole split.

Inputs (produced in-pod by run_c3_c4_pod.sh via run_lm_eval_v7 --limit 0 --log-samples):
  runs/vinf_causal/d_null_full.json   (base, --null-adapter)
  runs/vinf_causal/d_mc_full.json     (theta_mc offsets)

Integrated CANONICAL-REPRO GUARD: the first 400 doc_ids of the full split are the same
prompts as the canonical 400-slice (lm-eval --limit = first N, audit A-17), scored
deterministically -> per-item correctness must reproduce runs/vinf_causal/d_{null,mc}.json.
Hard-abort on >2% mismatch (wrong BASE/offsets) BEFORE the expensive gen-CoT stage.

Output: runs/vinf_causal/arbiter_arc_sets_fullsplit.json — same schema the gen script
consumes (gen_arbiter_arc_cot_pod.py --sets ... --subset base_wrong):
  mc_only       := base wrong AND mc right   (the adapter's fixes; no vinf leg here)
  fixed_by_both / vinf_only := []            (schema compat)
  base_right    := base right                (excluded by --subset base_wrong)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_arbiter_arc_sets import corr_map, extract_item

D = "runs/vinf_causal"
PATHS = {"null": os.path.join(D, "d_null_full.json"), "mc": os.path.join(D, "d_mc_full.json")}
OUT = os.path.join(D, "arbiter_arc_sets_fullsplit.json")


def canonical_guard(mnull_full, mmc_full):
    """Per-item repro check of the shared prefix vs the canonical 400-slice runs."""
    refs = {"null": (os.path.join(D, "d_null.json"), mnull_full),
            "mc": (os.path.join(D, "d_mc.json"), mmc_full)}
    for name, (rp, mfull) in refs.items():
        if not os.path.exists(rp):
            print(f"[guard] {rp} absent — skipping per-item canonical check ({name})", flush=True)
            continue
        mref = corr_map(rp)
        common = sorted(set(mref) & set(mfull))
        if not common:
            raise SystemExit(f"[guard] FATAL: no overlapping doc_ids with canonical {name}")
        mism = [d for d in common if (mref[d][0] or 0) != (mfull[d][0] or 0)]
        rate = len(mism) / len(common)
        acc_ref = sum((mref[d][0] or 0) for d in common) / len(common)
        acc_new = sum((mfull[d][0] or 0) for d in common) / len(common)
        print(f"[guard] {name}: {len(common)} shared items, mismatch {len(mism)} ({rate:.3f}); "
              f"acc ref={acc_ref:.4f} new={acc_new:.4f}", flush=True)
        if rate > 0.02:
            raise SystemExit(f"[guard] FATAL: {name} shared prefix does not reproduce the "
                             f"canonical run (mismatch {rate:.1%} > 2%) — wrong BASE/offsets? "
                             f"Aborting before the gen-CoT stage.")


def main():
    for k, p in PATHS.items():
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (run the C-3 cold-MC full-split evals first)")
    mnull, mmc = corr_map(PATHS["null"]), corr_map(PATHS["mc"])
    canonical_guard(mnull, mmc)
    ids = sorted(set(mnull) & set(mmc))

    sets = {"mc_only": [], "fixed_by_both": [], "vinf_only": [], "base_right": []}
    n_both_wrong = 0
    for did in ids:
        b, mc = mnull[did][0], mmc[did][0]
        if b == 1:
            sets["base_right"].append(did); continue
        if mc == 1:
            sets["mc_only"].append(did)
        else:
            n_both_wrong += 1

    items_all = [extract_item(mmc[did][1]) for did in ids]
    out = {"source": "runs/vinf_causal/d_{null,mc}_full.json (FULL ARC split, audit C-3)",
           "n_common": len(ids), "sets": sets, "n_both_wrong": n_both_wrong,
           "items_all": items_all}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    n_bw = len(ids) - len(sets["base_right"])
    print(f"[prep-full] n_common={len(ids)}  base_right={len(sets['base_right'])}  "
          f"base_wrong={n_bw} (mc_fixes={len(sets['mc_only'])}, both_wrong={n_both_wrong})", flush=True)
    print(f"[prep-full] wrote {OUT} ({len(items_all)} items)", flush=True)


if __name__ == "__main__":
    main()
