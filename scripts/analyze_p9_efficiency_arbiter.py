#!/usr/bin/env python3
"""P9 efficiency arbiter (queue #1, local €0).

Question (verdict.md §"Candidato positivo"): at activation-scale 0.5-0.75 the off-axis
offset keeps gsm8k acc ~= base (0.96-0.975) while the CoT is 27-36% SHORTER
(mean_len 150 -> 109 -> 96). Is that shorter CoT:
  (a) valid reasoning, just less verbose  -> genuine efficiency (win-win, prose pruning), OR
  (b) an off-axis shortcut that skips the arithmetic -> the same "skip-the-reasoning"
      mechanism as the cold-MC readout (NOT efficiency).

Method (forward-filter: guilty until an on-axis-style control absolves):
  - Condition on the INTERSECTION of correctness (base-correct AND scale-correct) per item,
    so the length delta is not confounded by "failed -> short".
  - Objective step metric: count gsm8k `<<a op b = c>>` calc annotations = explicit reasoning
    steps, independent of prose. Verify each step is arithmetically valid.
      * same #steps, fewer chars  -> prose pruning (efficiency real)
      * fewer #steps              -> arithmetic skipped (shortcut)
  - Emit the most-shortened items for the qualitative arbiter (read base vs scale0.75).

Input:  gsm8k_dose_lam0_gens.json (5 scales x 200 items, HF msap-p9-onaxis-20260709)
Output: prints tables; writes runs/p9_onaxis/efficiency_arbiter.json + top-shortened pairs.
"""
import json, re, sys, os

IN = sys.argv[1] if len(sys.argv) > 1 else "gsm8k_dose_lam0_gens.json"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "p9_onaxis")

STEP_RE = re.compile(r"<<([^>]+)>>")
SAFE_EXPR = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")
# annotation-robust: a binary arithmetic operation shown with its result, INSIDE OR OUTSIDE <<>>.
# Counts "a op b (op c ...) = d" so a completed CoT with no <<>> markers still scores its calcs.
CALC_RE = re.compile(r"\d[\d\.,]*\s*(?:[\+\-\*\/xX]\s*\d[\d\.,]*\s*)+=\s*\$?\d")

def parse_steps(gen):
    """Return list of (expr, stated_result, arith_ok) for each <<...>> annotation."""
    out = []
    for m in STEP_RE.finditer(gen):
        body = m.group(1)
        if "=" not in body:
            out.append((body, None, False)); continue
        lhs, rhs = body.rsplit("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        ok = False
        if SAFE_EXPR.match(lhs) and SAFE_EXPR.match(rhs):
            try:
                lv = eval(lhs, {"__builtins__": {}}, {})
                rv = eval(rhs, {"__builtins__": {}}, {})
                ok = abs(float(lv) - float(rv)) < 1e-6
            except Exception:
                ok = False
        out.append((body, rhs, ok))
    return out

def prose_chars(gen):
    """Chars OUTSIDE the <<...>> annotations = the natural-language / explanatory part."""
    return len(STEP_RE.sub("", gen))

def main():
    with open(IN, encoding="utf-8") as f:
        data = json.load(f)
    scales = [r["scale"] for r in data["results"]]
    n = data["config"]["n"]
    print(f"[cfg] scales={scales} n={n} max_new_tokens={data['config']['max_new_tokens']}", flush=True)

    # index per-item by scale
    by_scale = {}
    for r in data["results"]:
        items = r["per_item"]
        assert len(items) == n, f"scale {r['scale']}: {len(items)} != {n}"
        rows = []
        for it in items:
            steps = parse_steps(it["gen"])
            rows.append({
                "correct": it["correct"], "eos": it["eos"], "gen_len": it["gen_len"],
                "gen": it["gen"], "n_steps": len(steps),
                "n_steps_valid": sum(1 for _, _, ok in steps if ok),
                "n_calc": len(CALC_RE.findall(it["gen"])),  # annotation-robust reasoning-step count
                "prose": prose_chars(it["gen"]),
            })
        by_scale[r["scale"]] = rows
        print(f"[load] scale={r['scale']}: acc={r['acc']} mean_len={r['mean_gen_len']:.1f} "
              f"n_correct={r['n_correct']}", flush=True)

    base = by_scale[0.0]
    summary = {"config": data["config"], "scales": {}}
    print("\n=== ARBITER: intersection-conditioned (base-correct AND scale-correct) ===", flush=True)
    hdr = f"{'scale':>6} {'n_int':>6} {'gen_len':>18} {'n_steps':>16} {'steps_valid%':>13} {'prose_chars':>18}"
    print(hdr, flush=True)
    print(f"{'':>6} {'':>6} {'base->scale (Δ%)':>18} {'base->scale (Δ)':>16}", flush=True)

    for sc in scales:
        cur = by_scale[sc]
        inter = [i for i in range(n) if base[i]["correct"] == 1 and cur[i]["correct"] == 1]
        if not inter:
            continue
        def mean(rows, key, idxs): return sum(rows[i][key] for i in idxs) / len(idxs)
        bl = mean(base, "gen_len", inter); cl = mean(cur, "gen_len", inter)
        bs = mean(base, "n_steps", inter); cs = mean(cur, "n_steps", inter)
        bc = mean(base, "n_calc", inter);  cc = mean(cur, "n_calc", inter)
        bp = mean(base, "prose", inter);   cp = mean(cur, "prose", inter)
        # per-item validity fraction of steps (of items with >=1 step)
        vfrac = []
        for i in inter:
            if cur[i]["n_steps"] > 0:
                vfrac.append(cur[i]["n_steps_valid"] / cur[i]["n_steps"])
        vpct = 100 * sum(vfrac) / len(vfrac) if vfrac else float("nan")
        dlen = 100 * (cl - bl) / bl
        dstep = 100 * (cs - bs) / bs
        dcalc = 100 * (cc - bc) / bc
        dprose = 100 * (cp - bp) / bp
        print(f"{sc:>6.2f} {len(inter):>6} {bl:>7.1f}->{cl:>5.1f} ({dlen:>+5.1f}%) "
              f"<<>>:{bs:>4.2f}->{cs:>4.2f}({dstep:>+5.1f}%) calc:{bc:>4.2f}->{cc:>4.2f}({dcalc:>+5.1f}%) "
              f"valid{vpct:>5.1f}% prose{dprose:>+6.1f}%", flush=True)
        summary["scales"][sc] = {
            "n_intersect": len(inter), "base_gen_len": bl, "scale_gen_len": cl, "dlen_pct": dlen,
            "base_n_steps": bs, "scale_n_steps": cs, "dstep_pct": dstep,
            "base_n_calc": bc, "scale_n_calc": cc, "dcalc_pct": dcalc,
            "steps_valid_pct": vpct, "base_prose": bp, "scale_prose": cp, "dprose_pct": dprose,
        }

    # Also: on the FULL set, base step-validity for reference (is the base already ~100% valid?)
    def full_valid(rows):
        vf = [r["n_steps_valid"]/r["n_steps"] for r in rows if r["n_steps"] > 0]
        return 100*sum(vf)/len(vf) if vf else float("nan")
    print("\n=== step arithmetic validity, FULL set (all 200, per-item mean) ===", flush=True)
    for sc in scales:
        print(f"  scale={sc:.2f}: valid={full_valid(by_scale[sc]):.1f}%  "
              f"mean_steps={sum(r['n_steps'] for r in by_scale[sc])/n:.2f}", flush=True)

    # Top-shortened pairs on base ∩ scale0.75 for the qualitative arbiter
    TARGET = 0.75
    cur = by_scale[TARGET]
    inter = [i for i in range(n) if base[i]["correct"] == 1 and cur[i]["correct"] == 1]
    ranked = sorted(inter, key=lambda i: cur[i]["gen_len"] - base[i]["gen_len"])
    pairs = []
    for i in ranked[:12]:
        pairs.append({
            "item": i,
            "base_len": base[i]["gen_len"], "scale_len": cur[i]["gen_len"],
            "base_steps": base[i]["n_steps"], "scale_steps": cur[i]["n_steps"],
            "base_gen": base[i]["gen"], "scale_gen": cur[i]["gen"],
        })
    summary["top_shortened_base_vs_0.75"] = pairs

    os.makedirs(OUT_DIR, exist_ok=True)
    outp = os.path.join(OUT_DIR, "efficiency_arbiter.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[write] {outp}", flush=True)
    print(f"[write] {len(pairs)} top-shortened pairs for qualitative read", flush=True)

if __name__ == "__main__":
    main()
