"""Metric audit: how much of each published quantity is length normalisation?

Opened by FASE 1, which found that the same-param control's HellaSwag inducibility
falls from 69.5% (acc_norm) to 8.9% (acc). Since every "fraction of the lift" in the
paper uses acc_norm for ARC/HellaSwag and acc for TQA/Winogrande (method.tex:170-172),
the question is which published numbers move when the metric does. Three families of
quantity are checked:

  1. the four headline lifts (adapter - base)
  2. the inducibility fraction of every published control arm
  3. the L1 paired CI, via l1_paired_bootstrap_recovery_fraction.py --metric acc

TQA and Winogrande define no acc_norm, so they are unchanged by construction -- which
doubles as the pipeline check: if they came out different, the override would be
broken.

  python scripts/analyze_metric_sensitivity_audit.py
"""
import glob
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "runs/oq1_functional_axis/metric_sensitivity_audit.json"
MIN_LIFT_PP = 2.0  # below this the inducibility denominator is noise

R1_BASE = "runs/eval_bulletproof/R1_base/**/results_2026-06-29*.json"
R1_ALIGN = "runs/eval_bulletproof/R1_align/**/results_*.json"

# arm label -> (base json, contrastive json, [control jsons], control description)
CELLS = {
    "Align-LoRA r256 (R1, published spectrum anchor)": (
        R1_BASE, "runs/eval_bulletproof/R1_v7mc.json", [R1_ALIGN], "generic SFT, 1800x+ params"),
    "same-param plain-CE, Gemma (E1)": (
        "runs/e1_plain_ce/eval_base.json", "runs/e1_plain_ce/eval_contrastive_canonical.json",
        [f"runs/e1_plain_ce/eval_lifts_seed{s}.json" for s in (0, 1, 2)], "12,336 params, CE-plain"),
    "same-param plain-CE, Qwen (E6)": (
        "runs/e6_qwen_spine/eval_base.json", "runs/e6_qwen_spine/eval_mc.json",
        [f"runs/e6_qwen_spine/eval_plain_ce_seed{s}.final.json" for s in (0, 1, 2)],
        "6,192 params, CE-plain"),
}
TASK_METRICS = {"arc_challenge": ["acc_norm,none", "acc,none"],
                "hellaswag": ["acc_norm,none", "acc,none"],
                "truthfulqa_mc1": ["acc,none"], "winogrande": ["acc,none"]}


def results(path):
    if "*" in path:
        hits = sorted(glob.glob(path, recursive=True))
        if not hits:
            return None
        path = hits[0]
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def val(res, task, metric):
    return float(res[task][metric]) if res and task in res and metric in res[task] else None


def main():
    out = {"question": "which published quantities are length-normalisation effects?",
           "metric_policy": "paper uses acc_norm for ARC/HellaSwag, acc for TQA/Winogrande "
                            "(which define no acc_norm) -- method.tex:170-172",
           "cells": {}}

    for label, (bp, cp, ctlp, ctl_desc) in CELLS.items():
        base, con = results(bp), results(cp)
        if base is None or con is None:
            print(f"[skip] {label}: missing base/contrastive", flush=True)
            continue
        ctl = [r for r in (results(p) for p in ctlp) if r is not None]
        print(f"\n{'=' * 92}\n{label}   [control: {ctl_desc}, n_seeds={len(ctl)}]\n{'=' * 92}")
        print(f"  {'task':<16}{'metric':<10}{'lift pp':>9}{'ctl pp':>9}{'inducibility':>14}")
        rows = {}
        for task, metrics in TASK_METRICS.items():
            for m in metrics:
                b, c = val(base, task, m), val(con, task, m)
                if b is None or c is None:
                    continue
                cs = [val(r, task, m) for r in ctl]
                cs = [x for x in cs if x is not None]
                if not cs:
                    continue
                cm = float(np.mean(cs))
                lift, eff = (c - b) * 100, (cm - b) * 100
                ok = abs(lift) >= MIN_LIFT_PP
                rows[f"{task}/{m}"] = {"base": b, "contrastive": c, "control": cm,
                                       "lift_pp": lift, "control_pp": eff,
                                       "inducibility": (eff / lift) if ok else None,
                                       "readable": ok}
                shown = f"{eff / lift * 100:>13.1f}%" if ok else "   unreadable"
                print(f"  {task:<16}{m.replace(',none', ''):<10}{lift:>+9.2f}{eff:>+9.2f}{shown}")
        out["cells"][label] = {"control": ctl_desc, "n_seeds": len(ctl), "rows": rows}

    # --- the two metric-sensitive tasks, side by side across every arm ----------
    print(f"\n{'=' * 92}\nVERDICT -- inducibility under acc_norm vs acc\n{'=' * 92}")
    print(f"  {'arm':<40}{'task':<10}{'acc_norm':>10}{'acc':>12}{'delta':>10}")
    verdict = {}
    for label, cell in out["cells"].items():
        for task in ("arc_challenge", "hellaswag"):
            n = cell["rows"].get(f"{task}/acc_norm,none")
            a = cell["rows"].get(f"{task}/acc,none")
            if not n or not a:
                continue
            fn, fa = n["inducibility"], a["inducibility"]
            key = f"{label} | {task}"
            if fn is None or fa is None:
                verdict[key] = {"acc_norm": fn, "acc": fa, "delta": None,
                                "note": "unreadable under one metric (lift < 2pp)"}
                sn = f"{fn * 100:>9.1f}%" if fn is not None else "unreadable"
                sa = f"{fa * 100:>11.1f}%" if fa is not None else "  unreadable"
                print(f"  {label[:38]:<40}{task[:9]:<10}{sn:>10}{sa:>12}{'--':>10}")
                continue
            verdict[key] = {"acc_norm": fn, "acc": fa, "delta_pp": (fa - fn) * 100}
            print(f"  {label[:38]:<40}{task[:9]:<10}{fn * 100:>9.1f}%{fa * 100:>11.1f}%"
                  f"{(fa - fn) * 100:>+10.1f}")
    out["verdict_inducibility"] = verdict

    l1 = "runs/oq1_functional_axis/l1_paired_bootstrap.json"
    l1a = "runs/oq1_functional_axis/l1_paired_bootstrap_acc.json"
    if os.path.exists(l1) and os.path.exists(l1a):
        print(f"\n  L1 paired CI95 (200k reps, same items):")
        print(f"  {'task':<16}{'acc_norm (published)':<34}{'acc (robustness)':<34}")
        pairs = {}
        for a in json.load(open(l1, encoding="utf-8"))["per_task"]:
            b = next(x for x in json.load(open(l1a, encoding="utf-8"))["per_task"]
                     if x["task"] == a["task"])
            fa, fb = a["frac_ci95_paired"], b["frac_ci95_paired"]
            pairs[a["task"]] = {"acc_norm": {"point": a["frac_point"], "ci": fa,
                                             "p_gt_half": a["p_frac_gt_0.5"]},
                                "acc": {"point": b["frac_point"], "ci": fb,
                                        "p_gt_half": b["p_frac_gt_0.5"]},
                                "identical": abs(a["frac_point"] - b["frac_point"]) < 1e-12}
            print(f"  {a['task']:<16}"
                  f"{a['frac_point']:.4f} [{fa[0]:.3f}, {fa[1]:.3f}] P={a['p_frac_gt_0.5']:.3f}   "
                  f"{b['frac_point']:.4f} [{fb[0]:.3f}, {fb[1]:.3f}] P={b['p_frac_gt_0.5']:.3f}"
                  f"{'   [identical: no acc_norm defined]' if pairs[a['task']]['identical'] else ''}")
        out["l1_paired_both_metrics"] = pairs

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
