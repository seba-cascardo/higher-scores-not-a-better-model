"""Probe-as-readout DEPLOYED AS SELECTOR — the constructive corollary (B11, write-side).

The in-space probe (analyze_activation_probe_inspace) reports pair_auc(gold>distractor)=0.955.
This script asks the DEPLOYABLE question: if you USE that probe as the cold-MC answer
SELECTOR (argmax over options of held-out P(gold)), what is its ACCURACY — the +35 metric
(base acc_norm 0.505 / adapter 0.853) — read-only, on the same base activations, no offset
injected (so generation is byte-identical to base by construction; the probe never writes).

Fixes from the ClaraVT/Methodologist pass (do NOT regress these):
  - METRIC = selector ACCURACY over ALL items (not pair_auc on base-wrong). The +35 is acc.
  - PAIRED test = McNemar (probe-selector pick vs BASE cold-MC pick, per item) as PRIMARY —
    CI-separation of two marginals is underpowered at n=400. (McNemar vs the surface-stack
    selector needs the surface stack recomputed; flagged as follow-up, not done here.)
  - FLIP MATRIX vs base (base-correct kept / broken / base-wrong fixed / still-wrong): a PASS
    that secretly trades correct-for-wrong is NOT a recovery of the lift.
  - NULL in ACCURACY space (within-item label shuffle -> selector acc ~ 1/n_options ~0.25),
    NOT the AUC null (~0.5).
  - >=5 SEEDS for fold/PCA/C; report mean +/- across-seed SD (do not freeze on one seed).
  - Tier-1 caveat printed: held-out BY ITEM, feature-set (adapter cells) NOT held-out
    (those cells were correctness-selected during V7-mc training on ARC).

Ceiling: the supervised held-out surface SELECTOR accuracy (heldout_stack ~0.57).
We report it as the bar to beat in accuracy space; we do NOT recompute the surface stack here.

Reuses fit_heldout_scores / permuted_null from analyze_activation_probe_inspace (same fold,
preprocessing-inside-fold, AUC-tuned C). CPU-local, EUR 0.

Usage:
  python scripts/analyze_probe_as_selector.py \
      --acts runs/vinf_causal/coldmc_perhead_arc_challenge.pt \
      --anchor runs/vinf_causal/offaxis_vs_stack_arc_challenge.json \
      --seeds 0,1,2,3,4 --pca 200 --n-null 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_activation_probe_inspace import fit_heldout_scores  # noqa: E402

try:
    import torch
except Exception:
    torch = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Published anchors (accuracy space, ARC).
BASE_ACC = 0.505
ADAPTER_ACC = 0.853
SURFACE_CEILING_ACC = 0.57   # heldout_stack supervised surface selector


def selector_correct(proba, items):
    """Per item: argmax over its options of held-out P(gold). Returns dict
    item_idx -> 1/0 correct, plus the base_correct flag, aligned by item."""
    pc, bc = {}, {}
    for it in items:
        idxs = it["prompt_idxs"]
        s = np.array([proba[i] for i in idxs], dtype=np.float64)
        gold = it["target"]
        if gold >= len(s):
            continue
        pred = int(np.argmax(s))
        pc[it["item_idx"]] = int(pred == gold)
        bc[it["item_idx"]] = int(it["base_correct"])
    return pc, bc


def mcnemar_exact(b, c):
    """Exact two-sided McNemar on the discordant cells b, c (binomial)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def null_selector_acc(X, y, groups, items, Cs, seed, n_perm, max_iter, pca_k):
    """Within-item label shuffle -> fit held-out -> selector accuracy. ~1/n_options (~0.25)
    for 4-option ARC. If it clears base 0.505 the high-dim fit is leaking."""
    rng = np.random.default_rng(seed + 4242)
    accs = []
    for p in range(n_perm):
        y_perm = np.zeros(len(groups), dtype=np.int64)
        gold_perm = {}
        for it in items:
            idxs = it["prompt_idxs"]
            g = int(rng.integers(0, len(idxs)))
            y_perm[idxs[g]] = 1
            gold_perm[it["item_idx"]] = g
        proba_p = fit_heldout_scores(X, y_perm, groups, Cs, seed + p, max_iter, pca_k)
        ok = tot = 0
        for it in items:
            idxs = it["prompt_idxs"]
            s = np.array([proba_p[i] for i in idxs])
            if len(s) == 0:
                continue
            ok += int(int(np.argmax(s)) == gold_perm[it["item_idx"]]); tot += 1
        accs.append(ok / tot if tot else float("nan"))
        print(f"      null perm {p+1}/{n_perm}: selector_acc={accs[-1]:.3f}", flush=True)
    return accs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--anchor", type=Path, default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4", help=">=5 seeds for fold/PCA/C")
    ap.add_argument("--Cs", default="0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--pca", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if torch is None:
        raise SystemExit("torch required to load the .pt")

    st = torch.load(args.acts, weights_only=False, map_location="cpu")
    task = st["task"]
    A = st["acts_adapter"]                       # (n_prompts, n_cells, head_dim) — same-space
    X = A.reshape(A.shape[0], -1).numpy().astype(np.float64)
    meta = st["prompt_meta"]
    items = st["items"]
    y = np.array([m["is_gold"] for m in meta], dtype=np.int64)
    groups = np.array([m["item_idx"] for m in meta], dtype=np.int64)
    Cs = [float(x) for x in args.Cs.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 80, flush=True)
    print(f"PROBE-AS-SELECTOR | task={task} | feature_dim={X.shape[1]} (adapter cells) | "
          f"items={st['n_items']} | seeds={seeds}", flush=True)
    print("  CAVEAT (Tier-1): held-out BY ITEM, feature-set (adapter cells) NOT held-out "
          "(cells were correctness-selected during V7-mc training on ARC).", flush=True)
    print("=" * 80, flush=True)

    # multi-seed selector accuracy (held-out)
    per_seed_acc, last_pc, last_bc = [], None, None
    for sd in seeds:
        proba = fit_heldout_scores(X, y, groups, Cs, sd, args.max_iter, args.pca)
        pc, bc = selector_correct(proba, items)
        acc = float(np.mean(list(pc.values())))
        per_seed_acc.append(acc)
        print(f"  seed {sd}: probe-selector accuracy = {acc:.4f} (n={len(pc)})", flush=True)
        last_pc, last_bc = pc, bc
    acc_mean = float(np.mean(per_seed_acc))
    acc_sd = float(np.std(per_seed_acc))

    # flip matrix vs base (on the last seed's held-out picks)
    keys = sorted(set(last_pc) & set(last_bc))
    kept = sum(1 for k in keys if last_bc[k] == 1 and last_pc[k] == 1)
    broken = sum(1 for k in keys if last_bc[k] == 1 and last_pc[k] == 0)
    fixed = sum(1 for k in keys if last_bc[k] == 0 and last_pc[k] == 1)
    still = sum(1 for k in keys if last_bc[k] == 0 and last_pc[k] == 0)
    base_acc_here = float(np.mean([last_bc[k] for k in keys]))
    p_mcnemar = mcnemar_exact(broken, fixed)   # discordant: base-right/probe-wrong vs base-wrong/probe-right

    # null in accuracy space
    print(f"\n  fitting accuracy-space permutation null ({args.n_null} repeats)...", flush=True)
    null_accs = null_selector_acc(X, y, groups, items, Cs, seeds[0], args.n_null,
                                  args.max_iter, args.pca) if args.n_null > 0 else []
    null_mean = float(np.mean(null_accs)) if null_accs else float("nan")
    null_max = float(np.max(null_accs)) if null_accs else float("nan")
    null_clean = (not null_accs) or null_max <= BASE_ACC

    recovery_frac = (acc_mean - BASE_ACC) / (ADAPTER_ACC - BASE_ACC)

    # verdict (PASS = beats the surface ceiling in accuracy AND null is clean AND McNemar sig)
    beats_ceiling = acc_mean > SURFACE_CEILING_ACC
    pass_gate = beats_ceiling and null_clean and p_mcnemar < 0.05
    verdict = ("PASS (corollary lives): read-only probe-selector beats the surface ceiling in "
               "accuracy, McNemar-significant over base, null clean" if pass_gate else
               "FAIL/HEDGE (corollary stays diagnostic): does NOT clear the surface ceiling, "
               "or McNemar n.s., or null leaks -> probe stays evidence the signal EXISTS, not "
               "a deployable readout that recovers the lift")
    if not null_clean:
        verdict = (f"NULL CONTAMINATED: accuracy null reaches {null_max:.3f} (> base {BASE_ACC}) "
                   f"-> the {X.shape[1]}-dim fit leaks; the {acc_mean:.3f} headline is NOT trustworthy.")

    print(f"\n  === PROBE-AS-SELECTOR (ARC, held-out by item) ===", flush=True)
    print(f"      base cold-MC acc        : {BASE_ACC:.3f}", flush=True)
    print(f"      surface ceiling: {SURFACE_CEILING_ACC:.3f}  <- bar to beat", flush=True)
    print(f"      PROBE-SELECTOR acc      : {acc_mean:.4f} +/- {acc_sd:.4f}  (over {len(seeds)} seeds)", flush=True)
    print(f"      adapter acc             : {ADAPTER_ACC:.3f}", flush=True)
    print(f"      recovery fraction (probe-base)/(adapter-base) = {recovery_frac:.2f}", flush=True)
    print(f"\n  === FLIP MATRIX vs base (last seed; base_acc_here={base_acc_here:.3f}) ===", flush=True)
    print(f"      base-right kept   : {kept}", flush=True)
    print(f"      base-right BROKEN : {broken}   <- probe traded a correct for wrong", flush=True)
    print(f"      base-wrong FIXED  : {fixed}", flush=True)
    print(f"      base-wrong still  : {still}", flush=True)
    print(f"      McNemar exact (broken={broken} vs fixed={fixed}) p = {p_mcnemar:.2e}  "
          f"{'SIG (probe > base)' if p_mcnemar < 0.05 and fixed > broken else 'n.s. / not a gain'}", flush=True)
    print(f"      NULL selector acc : {null_mean:.3f} (max {null_max:.3f})  "
          f"{'OK ~0.25' if null_clean else '!!! LEAKING (> base) — headline NOT trustworthy'}", flush=True)
    print(f"\n  === VERDICT ===\n      {verdict}", flush=True)

    payload = dict(task=task, feature_dim=int(X.shape[1]), seeds=seeds,
                   base_acc=BASE_ACC, surface_ceiling=SURFACE_CEILING_ACC, adapter_acc=ADAPTER_ACC,
                   probe_selector_acc_mean=acc_mean, probe_selector_acc_sd=acc_sd,
                   recovery_fraction=recovery_frac,
                   flip_matrix=dict(kept=kept, broken=broken, fixed=fixed, still=still),
                   mcnemar_p=p_mcnemar, null_acc_mean=null_mean, null_acc_max=null_max,
                   null_clean=null_clean, verdict=verdict)
    outp = args.out or (args.acts.parent / f"probe_as_selector_{task}.json")
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {outp}", flush=True)


if __name__ == "__main__":
    main()
