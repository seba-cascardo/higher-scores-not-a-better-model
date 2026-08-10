#!/usr/bin/env python
"""O3-02: split GPQA's base-CoT `unparsed` into TRUNCATED and MALFORMED.

The paper cites base GPQA CoT accuracy of 0.7071 with 31 of 198 unparsed, and that split
has never been auditable: the artifact kept no token counts, and `parse_answer` ends in a
"lone letter in the last 100 chars" fallback. A parser with a fallback cannot measure its
own fallback (B57/B58) -- so truncation has to be decided by the TOKEN COUNT.

The re-run (2026-08-07, patched harness) persists `n_tokens`, `hit_cap` and
`finish_reason` per trace, which is what makes this script possible.

The number to look for is not `n_unparsed`. It is how many items the parser RESCUED: a
trace that ran out of budget mid-sentence, whose prose happened to contain a letter, and
that therefore entered the accuracy as if it were an answer.

Local, CPU, no download beyond the artifact itself.
"""
import argparse
import json
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="runs/vinf_causal/arbiter_gpqa_cot_traces.json")
    ap.add_argument("--reference", default="runs/vinf_causal/arbiter_gpqa_cot.json",
                    help="the artifact the paper cites, for the side-by-side")
    ap.add_argument("--out", default="outputs/o3_02/gpqa_truncation_audit.json")
    args = ap.parse_args()

    d = json.load(open(args.traces, encoding="utf-8"))
    rows, summ = d["results"], d["summary"]
    n = len(rows)
    cap = summ["max_new_tokens"]

    if "n_tokens" not in rows[0]:
        raise SystemExit("[audit] ABORT: el artefacto no trae n_tokens -- es una corrida "
                         "pre-parche y no puede auditarse. Re-generar con el harness actual.")

    def capped(r) -> bool:
        # By the count, never by the parser. `hit_cap` is written by the harness from the
        # generator's own token count; recomputed here so the file is self-checking.
        toks = r["n_tokens"] if isinstance(r["n_tokens"], list) else [r["n_tokens"]]
        return any(t >= cap - 2 for t in toks)

    parsed = [r for r in rows if r.get("pred_letter") is not None]
    unparsed = [r for r in rows if r.get("pred_letter") is None]
    trunc = [r for r in rows if capped(r)]

    # The 2x2 that the old artifact could not build.
    cells = {
        "truncated_unparsed": [r for r in unparsed if capped(r)],
        "truncated_parsed": [r for r in parsed if capped(r)],
        "complete_unparsed": [r for r in unparsed if not capped(r)],
        "complete_parsed": [r for r in parsed if not capped(r)],
    }
    # Rescued = truncated, yet given a letter, and given it by the FALLBACK rule.
    rescued = [r for r in cells["truncated_parsed"]
               if r.get("parse_rule") == "fallback_tail_letter"]
    rescued_correct = [r for r in rescued if r.get("base_cot_correct")]

    n_correct = sum(1 for r in rows if r.get("base_cot_correct"))
    acc = n_correct / n

    # Bounds: every truncated item is an item whose answer we do not have. Floor counts
    # them all wrong, ceiling counts them all right. The published point estimate sits
    # inside this interval and its width IS the cost of the cap.
    n_trunc = len(trunc)
    n_correct_complete = sum(1 for r in rows if r.get("base_cot_correct") and not capped(r))
    floor = n_correct_complete / n
    ceil = (n_correct_complete + n_trunc) / n

    print("=" * 66)
    print(f"n={n}  cap={cap}  acc_publicada_en_este_run={acc:.4f}")
    print(f"truncados (por CONTEO): {n_trunc} ({n_trunc/n:.1%})   unparsed: {len(unparsed)}")
    print("-" * 66)
    print(f"  truncado & unparsed : {len(cells['truncated_unparsed']):>4}")
    print(f"  truncado & PARSEADO : {len(cells['truncated_parsed']):>4}   <- respuesta sobre traza cortada")
    print(f"     de esos, por FALLBACK: {len(rescued):>4}   correctos: {len(rescued_correct)}")
    print(f"  completo & unparsed : {len(cells['complete_unparsed']):>4}   <- malformado de verdad")
    print(f"  completo & parseado : {len(cells['complete_parsed']):>4}")
    print("-" * 66)
    lo, hi = wilson(n_correct, n)
    print(f"acc puntual   : {acc:.4f}  Wilson95 [{lo:.4f}, {hi:.4f}]")
    print(f"acc BOUNDS    : [{floor:.4f}, {ceil:.4f}]  (ancho {ceil-floor:.4f} = costo del cap)")

    ref = None
    if Path(args.reference).exists():
        rd = json.load(open(args.reference, encoding="utf-8"))["summary"]
        ref = {"acc": rd.get("base_cot_acc_all"), "n_unparsed": rd.get("n_unparsed"),
               "max_new_tokens": rd.get("max_new_tokens"), "n": rd.get("n")}
        print("-" * 66)
        print(f"artefacto citado por el paper: acc={ref['acc']:.4f} unparsed={ref['n_unparsed']} "
              f"(n={ref['n']}, cap={ref['max_new_tokens']})")
        print(f"delta acc: {acc - ref['acc']:+.4f}")
        inside = floor <= ref["acc"] <= ceil
        print(f"el 0.7071 del paper cae dentro de los bounds de esta corrida: {inside}")

    out = {"n": n, "cap": cap, "acc": acc, "acc_wilson95": [lo, hi],
           "n_truncated": n_trunc, "n_unparsed": len(unparsed),
           "cells": {k: len(v) for k, v in cells.items()},
           "n_rescued_by_fallback": len(rescued),
           "n_rescued_by_fallback_correct": len(rescued_correct),
           "rescued_doc_ids": sorted(r["doc_id"] for r in rescued),
           "acc_bounds": [floor, ceil], "reference": ref}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("=" * 66)
    print(f"-> {args.out}")
    print("READ: `n_rescued_by_fallback` es el numero que el artefacto viejo no podia dar. "
          "Cada uno es un item sin respuesta que entro a la accuracy con la letra que su "
          "prosa cortada contenia. Si es 0, el 0.7071 no esta contaminado por truncacion y "
          "el unparsed es malformacion genuina.")


if __name__ == "__main__":
    main()
