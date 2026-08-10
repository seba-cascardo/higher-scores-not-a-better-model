"""U-08b — how much of the lift does the probe direction WRITE, on-axis?

LOCAL, CPU, EUR0. Reads the per-item lm-eval samples produced by
`scripts/run_u08b_probe_pod.sh` and applies the restriction the pod run could not:

  *** THE PROBE WAS FIT ON ARC ITEMS 0-399. ***

lm-eval's `--limit N` takes the FIRST N items, so the pod ran the FULL split
(`--limit 0`, n=1172) and the `doc_id >= 400` filter lands here. The blob states
the constraint itself, in runs/vinf_causal/mc_probe_offset.pt config.eval_constraint:
  "MUST be scored on ARC items >= 400 (probe fit on the rest)"

Recovery fraction (the U-08b number):
    frac = (acc_arm - acc_null) / (acc_mc - acc_null)      acc_norm, doc_id >= 400

Reported per arm: n, accuracy, frac, paired bootstrap CI of frac, exact McNemar
against null. Plus, as a DIAGNOSTIC only, the same computation restricted to the
contaminated slice 0-399 -- if the probe is overfit, its slice number is inflated
relative to its held-out one. That contrast is a sanity check on the fit, not the
result, and it is labelled as such in the output.

Paired tests, bootstrap and Wilson are IMPORTED from analyze_onaxis_significance
rather than reimplemented, so the two on-axis analyses share one instrument.

Provenance (B65): lm-eval leaves model_sha empty but stamps the cache path in
config.model_args, so the snapshot is recovered from there and written to the
output. If the arms do not share one snapshot the script REFUSES to compute --
the paper is protocol-bound to chat_template.jinja and the two known snapshots
differ in it.

Usage:
    python scripts/analyze_u08b_probe_recovery.py --dir runs/u08b_probe
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Works both as `python -m scripts.analyze_u08b_probe_recovery` and as
# `python scripts/analyze_u08b_probe_recovery.py` -- the runbook documents the
# second form, and without this it dies on the import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_onaxis_significance import (  # noqa: E402
    load_accnorm,
    mcnemar_exact,
    paired_bootstrap_frac,
    wilson_ci,
)

ARMS = ["null", "mc", "wknow", "probe"]
SNAP_RE = re.compile(r"snapshots/([0-9a-f]{6,40})")


def snapshot_of(path):
    """lm-eval json -> the model snapshot, recovered from config.model_args (B65)."""
    d = json.load(open(path, encoding="utf-8"))
    cfg = d.get("config", {})
    m = SNAP_RE.search(str(cfg.get("model_args", "")))
    return m.group(1) if m else None


def arm_stats(null, arm, mc, ids, label, n_boot, seed):
    """One arm on one item set. Returns a dict; prints as it goes."""
    n = len(ids)
    acc_n = sum(null[i] for i in ids) / n
    acc_a = sum(arm[i] for i in ids) / n
    acc_m = sum(mc[i] for i in ids) / n
    lift = acc_m - acc_n
    frac = (acc_a - acc_n) / lift if abs(lift) > 1e-9 else float("nan")

    bc, cc, p = mcnemar_exact(arm, null, ids)
    lo_acc, hi_acc = wilson_ci(int(round(acc_a * n)), n)
    print(f"    [{label}] bootstrap CI of the recovery fraction ...", flush=True)
    lo_f, hi_f = paired_bootstrap_frac(null, arm, mc, ids, n_boot=n_boot, seed=seed)

    return {
        "n": n,
        "acc_null": acc_n,
        "acc_arm": acc_a,
        "acc_mc": acc_m,
        "mc_lift_pp": 100 * lift,
        "recovery_frac": frac,
        "recovery_pct": 100 * frac,
        "recovery_ci95": [100 * lo_f, 100 * hi_f],
        "acc_wilson_ci95": [lo_acc, hi_acc],
        "mcnemar": {"arm_right_null_wrong": bc, "arm_wrong_null_right": cc,
                    "discordant": bc + cc, "p_two_sided": p},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/u08b_probe",
                    help="dir with d_{null,mc,wknow,probe}_full.json")
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--min-doc-id", type=int, default=400,
                    help="the probe fit boundary; items BELOW this are contaminated")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-snapshot-mismatch", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    out_path = Path(args.out) if args.out else d / "u08b_probe_recovery.json"

    # --- load, and refuse to compute on a broken comparand --------------------
    per_item, snaps, missing = {}, {}, []
    for arm in ARMS:
        p = d / f"d_{arm}_full.json"
        if not p.exists():
            missing.append(arm)
            continue
        acc = load_accnorm(p, task=args.task)
        if acc is None:
            raise SystemExit(f"FATAL: {p} has no samples[{args.task}] -- was it run "
                             f"with --log-samples? (it is on by default)")
        per_item[arm] = acc
        snaps[arm] = snapshot_of(p)
        print(f"  loaded {arm:6s} n={len(acc):5d}  snapshot={snaps[arm]}", flush=True)

    for req in ("null", "mc"):
        if req in missing:
            raise SystemExit(f"FATAL: the {req} arm is missing -- it is the comparand, "
                             f"not an optional arm.")
    if missing:
        print(f"  WARN arms absent: {', '.join(missing)}", flush=True)

    uniq = {s for s in snaps.values() if s}
    if len(uniq) > 1:
        msg = (f"FATAL: arms span {len(uniq)} snapshots {sorted(uniq)}. The paper is "
               f"protocol-bound to chat_template.jinja and the known snapshots differ "
               f"in it, so a cross-snapshot comparison measures template + arm at once.")
        if not args.allow_snapshot_mismatch:
            raise SystemExit(msg)
        print("  WARN " + msg, flush=True)
    snapshot = sorted(uniq)[0] if uniq else None

    # --- item sets ------------------------------------------------------------
    common = sorted(set.intersection(*(set(v) for v in per_item.values())))
    held = [i for i in common if i >= args.min_doc_id]
    slice_ = [i for i in common if i < args.min_doc_id]
    print(f"\n  common items: {len(common)}   held-out (>={args.min_doc_id}): {len(held)}"
          f"   contaminated slice: {len(slice_)}", flush=True)
    if not held:
        raise SystemExit("FATAL: no items at or above the probe fit boundary.")

    null, mc = per_item["null"], per_item["mc"]
    results = {"held_out": {}, "contaminated_slice_DIAGNOSTIC_ONLY": {}}

    print(f"\n=== HELD-OUT (doc_id >= {args.min_doc_id}) -- THE U-08b NUMBERS ===", flush=True)
    for arm in [a for a in ("wknow", "probe") if a in per_item]:
        print(f"  arm {arm}:", flush=True)
        results["held_out"][arm] = arm_stats(
            null, per_item[arm], mc, held, f"{arm}/held", args.n_boot, args.seed)

    print(f"\n=== SLICE 0-{args.min_doc_id - 1} -- DIAGNOSTIC, NOT A RESULT ===", flush=True)
    if slice_:
        for arm in [a for a in ("wknow", "probe") if a in per_item]:
            print(f"  arm {arm}:", flush=True)
            results["contaminated_slice_DIAGNOSTIC_ONLY"][arm] = arm_stats(
                null, per_item[arm], mc, slice_, f"{arm}/slice", args.n_boot, args.seed)

    # --- report ---------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print(f"U-08b -- on-axis WRITE, held out from the probe fit (n={len(held)})", flush=True)
    print("=" * 78, flush=True)
    hb = results["held_out"]
    any_arm = next(iter(hb.values()), None)
    if any_arm:
        print(f"  base (null) acc_norm : {any_arm['acc_null']:.4f}", flush=True)
        print(f"  mc          acc_norm : {any_arm['acc_mc']:.4f}"
              f"   lift = {any_arm['mc_lift_pp']:+.2f} pp", flush=True)
    for arm, r in hb.items():
        m = r["mcnemar"]
        print(f"\n  {arm}: acc_norm {r['acc_arm']:.4f}  "
              f"recovers {r['recovery_pct']:+.2f}% of the lift "
              f"[{r['recovery_ci95'][0]:+.2f}, {r['recovery_ci95'][1]:+.2f}]", flush=True)
        print(f"        McNemar vs null: {m['arm_right_null_wrong']}/{m['arm_wrong_null_right']} "
              f"discordant={m['discordant']}  p={m['p_two_sided']:.4g}", flush=True)
    sb = results["contaminated_slice_DIAGNOSTIC_ONLY"]
    if sb:
        print("\n  --- diagnostic (fit slice; inflation here = overfit, not a result) ---",
              flush=True)
        for arm, r in sb.items():
            hd = hb.get(arm, {}).get("recovery_pct", float("nan"))
            print(f"  {arm}: slice {r['recovery_pct']:+.2f}%  vs held-out {hd:+.2f}%"
                  f"   delta {r['recovery_pct'] - hd:+.2f} pp", flush=True)

    payload = {
        "task": args.task,
        "min_doc_id": args.min_doc_id,
        "n_held_out": len(held),
        "n_contaminated_slice": len(slice_),
        "model_snapshot": snapshot,
        "arm_snapshots": snaps,
        "arms_missing": missing,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "eval_constraint": "MUST be scored on ARC items >= 400 (probe fit on the rest)",
        "reading_fixed_in_advance": (
            "The wknow arm here is a NEW measurement under a declared protocol, not a "
            "re-test of the published 6.4%. If it differs, three causes are confounded "
            "(chat template, n 400->772, real effect) and this design cannot attribute "
            "it: the finding is 'the quantity is protocol-sensitive', never 'the 6.4% "
            "was wrong'."
        ),
        **results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
