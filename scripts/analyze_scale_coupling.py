#!/usr/bin/env python3
"""E8 — are the scoring lift and the length compression one gain or two?

The offset behaves as a monotone dial on two very different read-outs: cold-MC
scoring (the +35 headline) and generation length (755 -> 90 tokens, E7).
Monotonicity alone says nothing -- both are monotone in the scale by
construction, so a rank correlation between them is guaranteed and vacuous.

What discriminates is the SHAPE of each curve against the same dial. Anchoring
each read-out at scale=0 -> 0 and scale=1 -> 1 puts them on a common axis:

    f(s) = [v(s) - v(0)] / [v(1) - v(0)]          (length inverted to co-orient)

If one mechanism drives both, the two curves track. If they separate, the dial
is carrying two gains.

Per scale cell this reads:

  * scoring -- the per-item log-likelihoods lm-eval already logs
    (``--log-samples`` defaults True), recomputing acc / acc_norm plus the
    quantity acc_norm actually thresholds: the length-normalised margin between
    the gold option and the best distractor. The margin is the point --
    accuracy is bounded by chance below and 1.0 above, so a flat accuracy cell
    cannot distinguish "the mechanism saturated" from "the readout ran out of
    room". The margin has no ceiling.

  * generation -- mean length per scale, plus (for gsm8k) reasoning accuracy,
    which tests whether the brevity is what causes the reasoning damage.

CIs on f(s) come from a bootstrap that is PAIRED across cells: the same items
are resampled in every cell, because the cells share their item set and only
differ in the offset scale. Bootstrapping each cell independently would inflate
the interval with between-item variance that the contrast never sees.

Read-only on its inputs. Writes only under --out-dir.

Usage (local, CPU, no model needed):

    python scripts/analyze_scale_coupling.py \
        --scoring-dir runs/e8_scale_coupling/dose_sweep \
        --gen-json runs/e8_scale_coupling/dose_sweep/gsm8k_scale_sweep.json \
        --out-dir runs/e8_scale_coupling/positive
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

# The tables below carry non-ASCII (arrows, R^2); a Windows console defaults to
# cp1252 and would raise on the final print AFTER the artifacts are already on
# disk -- a confusing failure for a run that actually succeeded.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cells are named either `..._scale_<s>.json` (E7 chain) or `acc_norm_s<s>.json`
# (dose sweep). Both encode the scale in the filename; neither stores it inside.
_SCALE_RE = re.compile(r"(?:scale_|_s)(-?\d+(?:\.\d+)?)\.json$")


def cell_scale(path: str) -> float | None:
    m = _SCALE_RE.search(os.path.basename(path))
    return float(m.group(1)) if m else None


def load_scoring_cell(path: str, task: str | None) -> dict:
    """Per-item accuracy and margins from one lm-eval artifact."""
    d = json.load(open(path, encoding="utf-8"))
    samples = d.get("samples") or {}
    if not samples:
        raise SystemExit(
            f"FATAL: {path} has no 'samples' -- it ran with --no-log-samples, so the "
            "per-item log-likelihoods this analysis needs do not exist."
        )
    key = task if task in samples else sorted(samples)[0]
    if task and task not in samples:
        raise SystemExit(f"FATAL: task '{task}' not in {path} (has: {sorted(samples)})")

    per_item = {}
    for r in samples[key]:
        lls = [resp[0] for resp in r["filtered_resps"]]
        # lm-eval's acc_norm divides each continuation's loglik by its length in
        # characters; arguments[i] is (context, continuation).
        lens = [max(len(arg[1]), 1) for arg in r["arguments"]]
        norm = [ll / L for ll, L in zip(lls, lens)]
        gold = int(r["target"])

        def margin(v: list[float]) -> float:
            return v[gold] - max(x for i, x in enumerate(v) if i != gold)

        per_item[r["doc_id"]] = {
            "acc": float(max(range(len(lls)), key=lls.__getitem__) == gold),
            "acc_norm": float(max(range(len(norm)), key=norm.__getitem__) == gold),
            "margin_norm": margin(norm),
            "acc_norm_reported": r.get("acc_norm"),
        }
    return {"task": key, "per_item": per_item}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / max(len(xs), 1)


def pct(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return float("nan")
    i = min(int(p / 100 * len(sorted_xs)), len(sorted_xs) - 1)
    return sorted_xs[i]


def paired_bootstrap_f(by_scale: dict[float, list[float]], lo: float, hi: float,
                       iters: int, seed: int, invert: bool = False) -> dict:
    """Bootstrap f(s) = [v(s)-v(lo)] / [v(hi)-v(lo)], resampling items PAIRED.

    by_scale maps scale -> per-item values, all aligned to the same item order.
    """
    scales = sorted(by_scale)
    if lo not in by_scale or hi not in by_scale:
        return {}
    n = len(by_scale[lo])
    sign = -1.0 if invert else 1.0
    draws: dict[float, list[float]] = {s: [] for s in scales}
    rng = random.Random(seed)
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        m = {s: sign * mean(by_scale[s][i] for i in idx) for s in scales}
        den = m[hi] - m[lo]
        if den == 0:
            continue
        for s in scales:
            draws[s].append((m[s] - m[lo]) / den)
    out = {}
    for s in scales:
        d = sorted(draws[s])
        point = (sign * mean(by_scale[s]) - sign * mean(by_scale[lo])) / \
                (sign * mean(by_scale[hi]) - sign * mean(by_scale[lo]))
        out[s] = {"f": point, "ci": [pct(d, 2.5), pct(d, 97.5)], "draws": d}
    return out


def affine_fit(xs: list[float], ys: list[float]) -> dict:
    """Least-squares y = a + b*x with R^2. A rigid push predicts affine."""
    n = len(xs)
    if n < 3:
        return {"a": None, "b": None, "r2": None, "max_abs_resid": None}
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in resid) / ss_tot if ss_tot else float("nan")
    return {"a": a, "b": b, "r2": r2, "max_abs_resid": max(abs(r) for r in resid)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scoring-dir", required=True, help="dir of lm-eval scale cells")
    ap.add_argument("--scoring-glob", default="*.json")
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--gen-json", default=None,
                    help="gsm8k_scale_sweep.json or gen_free_scale_pod output")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-lo", type=float, default=0.0)
    ap.add_argument("--anchor-hi", type=float, default=1.0)
    ap.add_argument("--eos-floor", type=float, default=0.5,
                    help="pre-registered: cells below this eos_rate are excluded "
                         "from the coupling verdict (they measure a broken model)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- scoring cells ------------------------------------------------------
    paths = [p for p in sorted(glob.glob(os.path.join(args.scoring_dir, args.scoring_glob)))
             if cell_scale(p) is not None]
    if not paths:
        raise SystemExit(f"FATAL: no scale cells in {args.scoring_dir}")

    cells: dict[float, dict] = {}
    margins: dict[float, dict] = {}
    for i, p in enumerate(paths, 1):
        s = cell_scale(p)
        c = load_scoring_cell(p, args.task)
        pi = c["per_item"]
        margins[s] = pi
        rec = {"scale": s, "n": len(pi), "task": c["task"],
               "acc": mean(v["acc"] for v in pi.values()),
               "acc_norm": mean(v["acc_norm"] for v in pi.values()),
               "margin_norm": mean(v["margin_norm"] for v in pi.values()),
               "file": os.path.basename(p)}
        rep = [v["acc_norm_reported"] for v in pi.values() if v["acc_norm_reported"] is not None]
        if rep:
            rec["recompute_delta"] = abs(mean(rep) - rec["acc_norm"])
            if rec["recompute_delta"] > 1e-6:
                print(f"  !! scale={s:+.2f}: recomputed acc_norm off by "
                      f"{rec['recompute_delta']:.2e} — the margin definition does not "
                      f"match the metric it should explain.", flush=True)
        cells[s] = rec
        print(f"[{i}/{len(paths)}] scale={s:+.2f} n={rec['n']} acc={rec['acc']:.4f} "
              f"acc_norm={rec['acc_norm']:.4f} margin={rec['margin_norm']:+.5f}", flush=True)

    # align per-item margins across cells on the shared doc_id set
    common = sorted(set.intersection(*(set(m) for m in margins.values())))
    marg_by_s = {s: [margins[s][d]["margin_norm"] for d in common] for s in margins}
    print(f"aligned on {len(common)} shared items across {len(margins)} cells", flush=True)

    # --- generation curve ---------------------------------------------------
    gen: dict[float, dict] = {}
    gen_len_items: dict[float, list[float]] = {}
    if args.gen_json and os.path.exists(args.gen_json):
        gd = json.load(open(args.gen_json, encoding="utf-8"))
        rows = gd if isinstance(gd, list) else (gd.get("results") or [])
        for r in rows if isinstance(rows, list) else []:
            if not (isinstance(r, dict) and "scale" in r):
                continue
            s = float(r["scale"])
            gen[s] = {"mean_gen_len": r.get("mean_gen_len"), "acc": r.get("acc"),
                      "eos_rate": r.get("eos_rate"), "n": r.get("n")}
            if r.get("per_item"):
                gen_len_items[s] = [float(x["gen_len"]) for x in r["per_item"]]
        print(f"gen cells: {sorted(gen)}", flush=True)

    # --- shapes on a common axis -------------------------------------------
    lo, hi = args.anchor_lo, args.anchor_hi
    f_score = paired_bootstrap_f(marg_by_s, lo, hi, args.bootstrap, args.seed)
    f_len = (paired_bootstrap_f(gen_len_items, lo, hi, args.bootstrap, args.seed + 1,
                                invert=True) if gen_len_items else {})

    interior = [s for s in sorted(set(f_score) & set(f_len)) if lo < s < hi]
    excluded = [s for s in interior
                if (gen.get(s, {}).get("eos_rate") or 1.0) < args.eos_floor]
    interior = [s for s in interior if s not in excluded]

    sep = {}
    for s in interior:
        # the two read-outs come from disjoint item sets, so their bootstrap
        # draws are independent and can be differenced draw-by-draw
        a, b = f_score[s]["draws"], f_len[s]["draws"]
        d = sorted(x - y for x, y in zip(a, b))
        sep[s] = {"point": f_score[s]["f"] - f_len[s]["f"],
                  "ci": [pct(d, 2.5), pct(d, 97.5)]}

    max_sep = max((abs(v["point"]) for v in sep.values()), default=None)
    excl_zero = any(v["ci"][0] > 0 or v["ci"][1] < 0 for v in sep.values())
    if max_sep is None:
        verdict_b = "NO DATA"
    elif max_sep >= 0.30 and excl_zero:
        verdict_b = "DECOUPLED (two gains)"
    elif max_sep <= 0.15:
        verdict_b = "COUPLED (one gain)"
    else:
        verdict_b = "GREY ZONE — no conclusion"
    # pre-registered direction: f_len runs ahead of f_score in the interior
    dir_ok = all(v["point"] < 0 for v in sep.values()) if sep else None

    # --- KR-D: does the brevity cause the reasoning damage? ----------------
    kr_d = None
    if gen and all(g.get("acc") is not None for g in gen.values()):
        gs = sorted(gen)
        L = {s: gen[s]["mean_gen_len"] for s in gs}
        A = {s: gen[s]["acc"] for s in gs}
        span = L[gs[0]] - L[gs[-1]]
        rows = []
        for s in gs:
            rows.append({"scale": s, "len": L[s], "acc": A[s],
                         "len_frac": (L[gs[0]] - L[s]) / span if span else None,
                         "acc_drop_pp": 100 * (A[gs[0]] - A[s])})
        # compression without damage, and damage without further compression
        comp_no_dmg = [r for r in rows if r["len_frac"] and r["len_frac"] >= 0.5
                       and r["acc_drop_pp"] <= 1.0]
        pairs = [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
        dmg_no_comp = [(p, q) for p, q in pairs
                       if q["len"] >= p["len"] - 0.02 * span and
                       q["acc_drop_pp"] - p["acc_drop_pp"] >= 5.0]
        kr_d = {"rows": rows,
                "compression_without_damage": comp_no_dmg,
                "damage_without_compression": [{"from": p["scale"], "to": q["scale"],
                                                "d_len": q["len"] - p["len"],
                                                "d_acc_pp": -(q["acc_drop_pp"] - p["acc_drop_pp"])}
                                               for p, q in dmg_no_comp],
                "verdict": ("BREVITY DOES NOT CAUSE THE DAMAGE (double dissociation)"
                            if comp_no_dmg and dmg_no_comp else
                            "not separated by these cells")}

    fits = {"margin_norm": affine_fit(sorted(cells), [cells[s]["margin_norm"] for s in sorted(cells)]),
            "acc_norm": affine_fit(sorted(cells), [cells[s]["acc_norm"] for s in sorted(cells)])}
    if gen:
        gs = sorted(gen)
        fits["gen_len"] = affine_fit(gs, [gen[s]["mean_gen_len"] for s in gs])

    out = {"cells": [cells[s] for s in sorted(cells)],
           "gen": gen, "affine_fits": fits,
           "f_score": {s: {k: v for k, v in d.items() if k != "draws"} for s, d in f_score.items()},
           "f_len": {s: {k: v for k, v in d.items() if k != "draws"} for s, d in f_len.items()},
           "separation": sep, "anchors": [lo, hi],
           "kr_b": {"max_abs_separation": max_sep, "verdict": verdict_b,
                    "predicted_direction_f_len_ahead": dir_ok,
                    "interior_scales": interior, "excluded_low_eos": excluded},
           "kr_d": kr_d,
           "provenance": {"scoring_dir": args.scoring_dir, "task": args.task,
                          "gen_json": args.gen_json, "bootstrap": args.bootstrap,
                          "seed": args.seed, "n_items_scoring": len(common)}}
    json.dump(out, open(os.path.join(args.out_dir, "scale_coupling.json"), "w",
                        encoding="utf-8"), indent=2)

    # --- readable report ----------------------------------------------------
    L = ["# E8 — scale coupling: scoring lift vs generation length", "",
         f"scoring `{args.scoring_dir}` task=`{args.task}` n={len(common)} · "
         f"gen `{args.gen_json}` · bootstrap {args.bootstrap} (paired)", "",
         "| scale | acc | acc_norm | margin_norm | gen len | gen acc | eos |",
         "|---|---|---|---|---|---|---|"]
    for s in sorted(set(cells) | set(gen)):
        c, g = cells.get(s), gen.get(s, {})
        L.append(f"| {s:+.2f} | {c['acc']:.4f} | {c['acc_norm']:.4f} | {c['margin_norm']:+.5f} |"
                 if c else f"| {s:+.2f} | — | — | — |")
        L[-1] += (f" {g.get('mean_gen_len'):.1f} | {g.get('acc'):.4f} | {g.get('eos_rate'):.3f} |"
                  if g else " — | — | — |")
    L += ["", "## Affine fits (a single rigid push predicts affine)", "",
          "| curve | slope | R2 | max abs residual |", "|---|---|---|---|"]
    for k, f in fits.items():
        if f["b"] is not None:
            L.append(f"| {k} | {f['b']:+.5g} | {f['r2']:.4f} | {f['max_abs_resid']:.5g} |")

    if sep:
        L += ["", f"## KR-B — shapes on a common axis (anchored {lo:+.2f}=0, {hi:+.2f}=1)", "",
              "| scale | f_score [95% CI] | f_len [95% CI] | separation [95% CI] |",
              "|---|---|---|---|"]
        for s in interior:
            fs, fl, sp = f_score[s], f_len[s], sep[s]
            L.append(f"| {s:+.2f} | {fs['f']:+.4f} [{fs['ci'][0]:+.4f}, {fs['ci'][1]:+.4f}] "
                     f"| {fl['f']:+.4f} [{fl['ci'][0]:+.4f}, {fl['ci'][1]:+.4f}] "
                     f"| {sp['point']:+.4f} [{sp['ci'][0]:+.4f}, {sp['ci'][1]:+.4f}] |")
        L += ["", f"**max |separation| = {max_sep:.4f} → {verdict_b}**",
              f"(pre-registered direction f_len ahead of f_score: "
              f"{'held' if dir_ok else 'NOT held'})"]
        if excluded:
            L.append(f"excluded for eos_rate < {args.eos_floor}: {excluded}")

    if kr_d:
        L += ["", "## KR-D — does the brevity cause the reasoning damage?", "",
              "| scale | gen len | frac of length range | gen acc | acc drop pp |",
              "|---|---|---|---|---|"]
        for r in kr_d["rows"]:
            lf = f"{r['len_frac']:.3f}" if r["len_frac"] is not None else "—"
            L.append(f"| {r['scale']:+.2f} | {r['len']:.1f} | {lf} | {r['acc']:.4f} "
                     f"| {r['acc_drop_pp']:+.2f} |")
        L.append("")
        L.append(f"**{kr_d['verdict']}**")
        for d in kr_d["damage_without_compression"]:
            L.append(f"- damage without compression: {d['from']:+.2f} → {d['to']:+.2f}, "
                     f"length {d['d_len']:+.1f} tok, accuracy {d['d_acc_pp']:+.2f} pp")
        for r in kr_d["compression_without_damage"]:
            L.append(f"- compression without damage: at {r['scale']:+.2f}, "
                     f"{100*r['len_frac']:.0f}% of the length range spent, "
                     f"accuracy {r['acc_drop_pp']:+.2f} pp")

    mp = os.path.join(args.out_dir, "READ_scale_coupling.md")
    open(mp, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("", flush=True)
    print("\n".join(L), flush=True)
    print(f"\nwrote {mp}", flush=True)


if __name__ == "__main__":
    main()
