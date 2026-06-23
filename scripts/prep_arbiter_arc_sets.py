"""Build the ARC arbiter sets from the B causal-offset per-item outputs.

Reads runs/vinf_causal/d_{null,mc,vinf}.json (lm-eval per-sample acc_norm),
computes the offset sets, and extracts self-contained items (question + labeled
choices + gold) into runs/vinf_causal/arbiter_arc_sets.json so the pod gen-CoT
run (gen_arbiter_arc_cot_pod.py) needs no dataset reload.

  mc_only       : base wrong, mc right, v_inf-offset wrong   (the 124 — the target)
  fixed_by_both : base wrong, both mc and v_inf right
  vinf_only     : base wrong, v_inf right, mc wrong
  base_right    : base already right (control)

€0 local. On a fresh pod, pull the d_*.json from HF first
(sebacascardo87/msap-vinf-causal-20260623) — run_arbiter_arc_cot_pod.sh does this.
"""
import json
import os

D = "runs/vinf_causal"
PATHS = {k: os.path.join(D, f"d_{k}.json") for k in ["null", "mc", "vinf"]}
OUT = os.path.join(D, "arbiter_arc_sets.json")


def load_arc_samples(path):
    obj = json.load(open(path, encoding="utf-8"))
    samples = obj.get("samples")
    if isinstance(samples, dict):
        for tk in samples:
            if "arc" in tk:
                return samples[tk]
    for tk in obj:
        if "arc" in tk and isinstance(obj[tk], list):
            return obj[tk]
    raise SystemExit(f"no arc_challenge per-sample list in {path}")


def corr_map(path):
    m = {}
    for i, s in enumerate(load_arc_samples(path)):
        did = s.get("doc_id", i)
        acc = s.get("acc_norm")
        if acc is None and isinstance(s.get("metrics"), dict):
            acc = s["metrics"].get("acc_norm")
        m[did] = (int(round(float(acc))) if acc is not None else None, s)
    return m


def extract_item(s):
    doc = s.get("doc", {}) if isinstance(s.get("doc"), dict) else {}
    return {"doc_id": s.get("doc_id"), "arc_id": doc.get("id"),
            "question": doc.get("question"), "choices": doc.get("choices"),
            "answerKey": doc.get("answerKey"), "gold": s.get("target", s.get("gold"))}


def main():
    for k, p in PATHS.items():
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (pull from HF msap-vinf-causal-20260623)")
    mnull, mmc, mvinf = corr_map(PATHS["null"]), corr_map(PATHS["mc"]), corr_map(PATHS["vinf"])
    ids = sorted(set(mnull) & set(mmc) & set(mvinf))

    sets = {"mc_only": [], "fixed_by_both": [], "vinf_only": [], "base_right": []}
    for did in ids:
        b, mc, vf = mnull[did][0], mmc[did][0], mvinf[did][0]
        if b == 1:
            sets["base_right"].append(did); continue
        if mc == 1 and vf == 0:
            sets["mc_only"].append(did)
        if mc == 1 and vf == 1:
            sets["fixed_by_both"].append(did)
        if mc == 0 and vf == 1:
            sets["vinf_only"].append(did)

    items_all = [extract_item(mmc[did][1]) for did in ids]
    out = {"source": "runs/vinf_causal/d_{null,mc,vinf}.json", "n_common": len(ids),
           "sets": sets, "items_all": items_all}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"[prep] n_common={len(ids)}  " + " ".join(f"{k}={len(v)}" for k, v in sets.items()))
    print(f"[prep] wrote {OUT} ({len(items_all)} items)")


if __name__ == "__main__":
    main()
