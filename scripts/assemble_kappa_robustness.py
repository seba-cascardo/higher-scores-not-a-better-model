"""Assemble the KAPPA robustness envelope for GATE-2 (local, post-pod, EUR0).

Reads the sweep of score_mc_kappa_pod.py outputs (runs/gate2/kappa_*.json across
wpred-source / k / alpha / beta / ridge configs) and reports, PER TASK, the ENVELOPE
of KAPPA's on-axis recovery so the paper can claim "no reconstructed on-axis KAPPA,
across any reasonable hyperparameter, recovers more than X% of the off-axis lift"
instead of resting on a single reconstructed point (the W#-D-NEW-2 defense).

recovery% = (kappa_acc_norm - base_acc_norm) / (offaxis_restricted_acc_norm - base_acc_norm)
            measured on the SAME cells (head-set parity; score_mc_kappa_pod aborts if not).

GATES applied here (read, do not tune):
  * a config whose sanity_gate_1_scaffold.status == WARNING is DROPPED from the
    envelope (broken scaffold => its numbers are silently wrong).
  * the wknow self-test config (KS6) must show |Δ| ~ 0 (near-identity); flagged if not.
  * any config with recovery >= the FLAG_RECOVERY fraction (default 0.50) of the
    off-axis lift is surfaced for scrutiny (would weaken "on-axis fails").

Usage:
  python scripts/assemble_kappa_robustness.py --glob "runs/gate2/kappa_*.json" \
      --out runs/gate2/kappa_robustness_envelope.json
Compile-check:  python -c "import ast;ast.parse(open('scripts/assemble_kappa_robustness.py').read())"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FLAG_RECOVERY = 0.50          # surface configs whose KAPPA recovery >= this fraction
W_KNOW_BASELINE = 0.064       # the canonical on-axis foil fraction (W_know, ARC)


def _recovery(m):
    """KAPPA recovery fraction of the off-axis (restricted) lift, acc_norm. None if
    the off-axis lift is ~0 (undefined ratio) or arms missing."""
    base = m.get("base", {}).get("acc_norm")
    kap = m.get("kappa", {}).get("acc_norm")
    off = (m.get("offaxis_restricted") or {}).get("acc_norm")
    if base is None or kap is None or off is None:
        return None, base, kap, off
    denom = off - base
    if abs(denom) < 1e-6:
        return None, base, kap, off
    return (kap - base) / denom, base, kap, off


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="runs/gate2/kappa_*.json")
    ap.add_argument("--flag-recovery", type=float, default=FLAG_RECOVERY)
    ap.add_argument("--out", type=Path, default=Path("runs/gate2/kappa_robustness_envelope.json"))
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    print("=" * 74, flush=True)
    print(f"KAPPA robustness envelope | {len(files)} files matched {args.glob!r}", flush=True)
    print("=" * 74, flush=True)
    if not files:
        raise SystemExit(f"[FATAL] no files matched {args.glob!r}. Run the pod sweep first "
                         f"(score_mc_kappa_pod.py across configs). This is a post-pod tool.")

    rows = []
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] unreadable {fp}: {e}", flush=True)
            continue
        task = d.get("task", "?")
        cfg = d.get("config", {})
        m = d.get("metrics", {})
        scaffold = (d.get("sanity_gate_1_scaffold") or {}).get("status")
        diag = d.get("subspace_diagnostics", {})
        rec, base, kap, off = _recovery(m)
        wpred = cfg.get("wpred_source")
        row = {"file": fp, "task": task, "wpred": wpred,
               "k_pred": cfg.get("k_pred"), "k_know": cfg.get("k_know"),
               "alpha": cfg.get("kappa_alpha"), "beta": cfg.get("kappa_beta"),
               "ridge_frac": cfg.get("ridge_frac"),
               "base_acc_norm": base, "kappa_acc_norm": kap, "offaxis_restricted_acc_norm": off,
               "kappa_recovery_frac": rec, "scaffold_status": scaffold,
               "cos_wpred_wknow_mean": diag.get("cos_wpred_wknow_mean"),
               "is_selftest_wknow": (wpred == "wknow")}
        rows.append(row)
        rec_s = f"{rec:+.3f}" if rec is not None else "n/a"
        print(f"  {task:16s} wpred={str(wpred):5s} a={row['alpha']} b={row['beta']} "
              f"kp={row['k_pred']} rg={row['ridge_frac']}: base={base} kappa={kap} "
              f"off_r={off} recovery={rec_s} scaffold={scaffold}", flush=True)

    # --- self-test check (KS6: wpred==wknow should be near-identity) -------------
    selftest = [r for r in rows if r["is_selftest_wknow"]]
    for r in selftest:
        if r["kappa_acc_norm"] is not None and r["base_acc_norm"] is not None:
            drift = r["kappa_acc_norm"] - r["base_acc_norm"]
            if abs(drift) > 0.02:
                print(f"  !!! SELF-TEST FAIL ({r['task']}): wpred==wknow Δacc_norm={drift:+.4f} "
                      f"(expected ~0, near-identity). Inspect Eq.3 codepath.", flush=True)

    # --- envelope per task (drop scaffold-WARNING and self-test configs) ---------
    envelope = {}
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        valid = [r for r in rows
                 if r["task"] == task and not r["is_selftest_wknow"]
                 and r["scaffold_status"] != "WARNING"
                 and r["kappa_recovery_frac"] is not None]
        dropped = [r["file"] for r in rows
                   if r["task"] == task and (r["is_selftest_wknow"]
                                             or r["scaffold_status"] == "WARNING"
                                             or r["kappa_recovery_frac"] is None)]
        if not valid:
            envelope[task] = {"n_valid": 0, "dropped": dropped,
                              "note": "no valid configs (all dropped)"}
            continue
        recs = [(r["kappa_recovery_frac"], r) for r in valid]
        rmax, rmax_row = max(recs, key=lambda t: t[0])
        rmin, rmin_row = min(recs, key=lambda t: t[0])
        flagged = [r["file"] for r in valid
                   if r["kappa_recovery_frac"] >= args.flag_recovery]
        envelope[task] = {
            "n_valid": len(valid), "n_dropped": len(dropped), "dropped": dropped,
            "kappa_recovery_max": rmax, "kappa_recovery_max_config": rmax_row["file"],
            "kappa_recovery_min": rmin, "kappa_recovery_min_config": rmin_row["file"],
            "kappa_recovery_all": sorted(r["kappa_recovery_frac"] for r in valid),
            "w_know_baseline_frac": (W_KNOW_BASELINE if task == "arc_challenge" else None),
            "configs_over_flag_recovery": flagged,
            "claim_supported": (rmax < args.flag_recovery),
            "claim": (f"No reconstructed on-axis KAPPA config recovers more than "
                      f"{rmax:.1%} of the off-axis lift on {task} "
                      f"(flag at {args.flag_recovery:.0%})."),
        }
        print(f"\n  [{task}] envelope: recovery in [{rmin:+.3f}, {rmax:+.3f}] over "
              f"{len(valid)} valid configs; flag(>={args.flag_recovery:.0%})="
              f"{len(flagged)}; claim_supported={envelope[task]['claim_supported']}",
              flush=True)

    payload = {"glob": args.glob, "flag_recovery": args.flag_recovery,
               "n_files": len(files), "rows": rows, "envelope": envelope}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  saved: {args.out}", flush=True)
    print("  next: report the per-task envelope max alongside W_know=6.4% (ARC); the "
          "GATE-2 head-1 'on-axis fails' claim rests on this envelope, not a single point.",
          flush=True)


if __name__ == "__main__":
    main()
