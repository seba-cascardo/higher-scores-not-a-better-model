"""Test 1 (COARSE / per-option) — decompose the cold-MC +35 lift into WHO moves:
does the contrastive adapter RAISE the correct option's loglik or SUPPRESS the
incorrect ones? And does the per-option Δloglik scale with option length (a proxy
for structural-token mass)? €0-local from existing samples (no per-token logprobs).

HYPOTHESIS under test (de-swamping): the adapter suppresses the probability mass of
structural/format tokens that swamp the cold loglik; the correct option resists
because it carries less swamping mass relative to its content signal. PREDICTION:
incorrect options drop MORE than the correct one (selective suppression), and the
flips (base-wrong→v7mc-right) are driven by incorrects collapsing, not the correct
rising. If instead the correct rises selectively → de-swamping is wrong, it's
'knowledge'. Length-correlation: more structural mass (longer option) → bigger drop.

CAVEAT: this is per-OPTION, not per-TOKEN. It localizes WHO moves, not WHICH tokens.
The token-level structural-vs-content split (Test 1 fine) needs a GPU re-extraction
of per-token logprobs — not available in standard lm-eval samples.

  MSAP_ROOT=. python scripts/analyze_coldmc_loglik_decomposition.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.environ.get("MSAP_ROOT", ".")
BASE_GLOB = os.path.join(ROOT, "runs/eval_bulletproof/R1_base/**/samples_arc_challenge_*.jsonl")
V7MC = os.path.join(ROOT, "runs/eval_bulletproof/R1_v7mc.json")
OUT = os.path.join(ROOT, "runs/derisk/test1_coldmc_loglik_decomp.json")


def _ll(resps_entry):
    # filtered_resps[opt] is [loglik, is_greedy]; base stores strings, v7mc floats
    return float(resps_entry[0])


def load_doc_map(samples):
    out = {}
    for s in samples:
        fr = s["filtered_resps"]
        ll = np.array([_ll(x) for x in fr], dtype=float)
        # option char-lengths for the length/acc_norm analysis (best-effort)
        lens = None
        doc = s.get("doc", {})
        ch = doc.get("choices")
        if isinstance(ch, dict) and "text" in ch:
            lens = np.array([len(t) for t in ch["text"]], dtype=float)
        out[s["doc_id"]] = {"ll": ll, "target": int(s["target"]), "lens": lens,
                            "hash": s.get("doc_hash")}
    return out


def main():
    bpath = glob.glob(BASE_GLOB, recursive=True)[0]
    base = load_doc_map([json.loads(l) for l in open(bpath)])
    v7 = load_doc_map(json.load(open(V7MC))["samples"]["arc_challenge"])
    ids = sorted(set(base) & set(v7))
    print(f"[test1] base {len(base)} | v7mc {len(v7)} | aligned {len(ids)}")
    mism = sum(1 for i in ids if base[i]["hash"] and v7[i]["hash"] and base[i]["hash"] != v7[i]["hash"])
    print(f"[test1] doc_hash mismatches: {mism} (should be 0)")

    dC, dW = [], []                  # Δloglik correct / mean-incorrect (all docs)
    dC_flip, dWwin_flip = [], []     # on flips: Δ correct / Δ of the base-winning wrong opt
    margin_base, margin_v7 = [], []
    dl_all, len_all = [], []         # per-option Δloglik vs length (correlation)
    acc_base = acc_v7 = accn_base = accn_v7 = 0
    n_flip = n = 0
    for i in ids:
        b, v = base[i], v7[i]
        if len(b["ll"]) != len(v["ll"]) or b["target"] != v["target"]:
            continue
        t = b["target"]; n += 1
        d = v["ll"] - b["ll"]
        wrong = [j for j in range(len(d)) if j != t]
        dC.append(d[t]); dW.append(float(np.mean(d[wrong])))
        margin_base.append(b["ll"][t] - max(b["ll"][j] for j in wrong))
        margin_v7.append(v["ll"][t] - max(v["ll"][j] for j in wrong))
        # acc (raw argmax) and acc_norm (loglik / char-len argmax)
        pb, pv = int(np.argmax(b["ll"])), int(np.argmax(v["ll"]))
        acc_base += (pb == t); acc_v7 += (pv == t)
        if b["lens"] is not None and v["lens"] is not None:
            pbn = int(np.argmax(b["ll"] / b["lens"])); pvn = int(np.argmax(v["ll"] / v["lens"]))
            accn_base += (pbn == t); accn_v7 += (pvn == t)
            for j in range(len(d)):
                dl_all.append(d[j]); len_all.append(b["lens"][j])
        if pb != t and pv == t:       # FLIP: base wrong -> v7mc right
            n_flip += 1
            dC_flip.append(d[t])
            dWwin_flip.append(d[pb])  # Δ of the option that WON wrongly in base
    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    res = {
        "n": n, "n_flip": n_flip,
        "acc_base": acc_base / n, "acc_v7mc": acc_v7 / n, "acc_lift": (acc_v7 - acc_base) / n,
        "acc_norm_base": accn_base / n if accn_base or accn_v7 else None,
        "acc_norm_v7mc": accn_v7 / n if accn_base or accn_v7 else None,
        "delta_loglik_correct_mean": m(dC),
        "delta_loglik_incorrect_mean": m(dW),
        "suppression_ratio_incorrect_over_correct": m(dW) / m(dC) if m(dC) != 0 else None,
        "margin_base_mean": m(margin_base), "margin_v7mc_mean": m(margin_v7),
        "flip_delta_correct_mean": m(dC_flip),
        "flip_delta_basewinner_mean": m(dWwin_flip),
        "len_corr_pearson_dloglik_vs_optlen": (
            float(np.corrcoef(dl_all, len_all)[0, 1]) if len(dl_all) > 2 else None),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print("\n=== Test 1 (per-option) — WHO moves under the contrastive adapter ===")
    print(f"  acc: base {res['acc_base']:.3f} -> v7mc {res['acc_v7mc']:.3f}  (lift {res['acc_lift']*100:+.1f}pp)")
    if res["acc_norm_base"] is not None:
        print(f"  acc_norm: base {res['acc_norm_base']:.3f} -> v7mc {res['acc_norm_v7mc']:.3f}  "
              f"(if lift survives here it's NOT pure length)")
    print(f"\n  Δloglik CORRECT   (mean): {res['delta_loglik_correct_mean']:+.2f}")
    print(f"  Δloglik INCORRECT (mean): {res['delta_loglik_incorrect_mean']:+.2f}")
    print(f"  → suppression ratio incorrect/correct: {res['suppression_ratio_incorrect_over_correct']}")
    print(f"     (>>1 or incorrect≪0 while correct≈0/+ → SUPPRESSION of wrongs = de-swamping)")
    print(f"\n  decision margin (correct − best wrong): base {res['margin_base_mean']:+.2f} -> v7mc {res['margin_v7mc_mean']:+.2f}")
    print(f"\n  FLIPS (base-wrong→v7mc-right): {n_flip}/{n}")
    print(f"     Δ correct on flips        : {res['flip_delta_correct_mean']:+.2f}")
    print(f"     Δ base-winner(wrong) flips: {res['flip_delta_basewinner_mean']:+.2f}")
    print(f"     → if base-winner ≪ correct: the flip is the WRONG option collapsing, not the correct rising")
    print(f"\n  Pearson(Δloglik, option char-len): {res['len_corr_pearson_dloglik_vs_optlen']}")
    print(f"     (strong negative → longer option dropped more = structural-mass/length-driven)")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
