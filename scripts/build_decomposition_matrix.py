"""E1 — per-item decomposition matrix (€0-local). Tests novel-claim 2: cold-scoring
MISRANKING, not missing knowledge. Does the per-item hack lift track gen-solvability
(CoT-accessible) rather than item difficulty / length artifacts?

Columns available €0-local now (from d_*.json + arbiter): hack_fixed, base/mc correct,
length_effect (acc vs acc_norm divergence), gen_solvable (base CoT), gen_len.
HOOKS (need E2-pod answer-only + a probe): surface_bias (choices-only), PMI_residual,
probe_score — documented join points, not yet computed.

  python scripts/build_decomposition_matrix.py
"""
import json
import os

import numpy as np

D = "runs/vinf_causal"
# Prefer the parser-fixed v2 arbiter when present (doc 128 parsed; n_unparsed 0).
# Same resolution rule as analyze_cold_mc_calibration.py / analyze_p4_base_arm_power.py.
_ARB_V2 = "runs/vinf_causal/arbiter_arc_cot_v2.json"
ARB = _ARB_V2 if os.path.exists(_ARB_V2) else "runs/vinf_causal/arbiter_arc_cot.json"
OUT = "runs/vinf_causal/decomposition_matrix_arc.json"


def load_arc(path):
    obj = json.load(open(path, encoding="utf-8"))
    smp = obj["samples"]
    for tk in smp:
        if "arc" in tk:
            return smp[tk]
    raise SystemExit(f"no arc samples in {path}")


def corr_map(samples, key):
    m = {}
    for i, s in enumerate(samples):
        did = s.get("doc_id", i)
        m[did] = {"acc": int(round(float(s.get("acc", 0)))),
                  "acc_norm": int(round(float(s.get("acc_norm", 0))))}
    return m


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    nul = corr_map(load_arc(f"{D}/d_null.json"), "n")
    mc = corr_map(load_arc(f"{D}/d_mc.json"), "m")
    cot = {r["doc_id"]: r for r in json.load(open(ARB, encoding="utf-8"))["results"]}
    ids = sorted(set(nul) & set(mc) & set(cot))

    rows = []
    for did in ids:
        b = nul[did]; m = mc[did]; c = cot[did]
        rows.append({
            "doc_id": did,
            "base_correct": b["acc_norm"],
            "mc_correct": m["acc_norm"],
            "hack_fixed": int(b["acc_norm"] == 0 and m["acc_norm"] == 1),
            "base_length_effect": int(b["acc"] != b["acc_norm"]),
            "gen_solvable": int(c.get("base_cot_correct", 0)),
            "gen_len": int(c.get("gen_len", 0)),
        })

    base_wrong = [r for r in rows if r["base_correct"] == 0]
    fixed = [r for r in rows if r["hack_fixed"] == 1]
    print(f"[E1] n={len(rows)}  base_wrong={len(base_wrong)}  hack_fixed={len(fixed)}")

    # E6 / novel-claim-2 correlations on the base_wrong subset (where the lift is defined)
    hv = [r["hack_fixed"] for r in base_wrong]
    gs = [r["gen_solvable"] for r in base_wrong]
    gl = [r["gen_len"] for r in base_wrong]
    le = [r["base_length_effect"] for r in base_wrong]
    cors = {
        "hack_fixed_vs_gen_solvable": spearman(hv, gs),
        "hack_fixed_vs_gen_len": spearman(hv, gl),
        "hack_fixed_vs_length_effect": spearman(hv, le),
    }
    # contrast: gen-solvable rate among fixed vs not-fixed (within base_wrong)
    fx = [r for r in base_wrong if r["hack_fixed"]]
    nfx = [r for r in base_wrong if not r["hack_fixed"]]
    gen_rate_fixed = np.mean([r["gen_solvable"] for r in fx]) if fx else float("nan")
    gen_rate_notfixed = np.mean([r["gen_solvable"] for r in nfx]) if nfx else float("nan")

    res = {"n": len(rows), "n_base_wrong": len(base_wrong), "n_fixed": len(fixed),
           "spearman_base_wrong": cors,
           "gen_solvable_rate": {"fixed": float(gen_rate_fixed), "not_fixed": float(gen_rate_notfixed)},
           "hooks_TODO": {
               "surface_bias": "choices-only correctness from score_mc_premise_pod.py (E2-pod answer-only)",
               "PMI_residual": "from score_mc_pmi.py --premise-file (E2)",
               "probe_score": "train logistic on runs/correctness_probe_postgen/postgen_arc_challenge.pt (verify cold-MC provenance first)"},
           "rows": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print("\n=== E1 (available columns; base_wrong subset) ===")
    for k, v in cors.items():
        print(f"  Spearman {k:32s} = {v:+.3f}")
    print(f"  gen_solvable rate: fixed {gen_rate_fixed:.3f} vs not_fixed {gen_rate_notfixed:.3f}")
    print("\nREAD: novel-claim-2 = the hack fixes CoT-SOLVABLE items. If fixed items are ~all "
          "gen_solvable AND not_fixed (still-wrong) are LESS solvable, the hack targets latent-but-"
          "knowable knowledge (misranking). gen_len corr ~0 supports kill-B (not amortized serial CoT).")
    print("HOOKS pending (E2-pod surface_bias/PMI + probe_score) — see hooks_TODO in output.")
    print(f"[E1] wrote {OUT}")


if __name__ == "__main__":
    main()
