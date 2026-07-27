"""R12 — freeze the item sets for the paired generation evaluation (LOCAL, CPU-only).

Pre-registration: docs/superpowers/specs/2026-07-27-paired-generation-preregistration.md

Every arm must see byte-identical items. The project has already paid for slice drift
(GPQA's enumerate() ids moving under a dataset update), so the items are frozen to
self-contained JSON here, once, and no pod script ever reloads a dataset.

  arc_full     1172  reused verbatim from the full-split arbiter sets -- NOT rebuilt,
                     so the 444 mc_only items stay exactly the ones KR1 is about
  mmlu_pro     1000  proportional stratified sample by category, seed 0
  hellaswag    1000  random sample, seed 0
  winogrande   1267  full validation split

Writes runs/paired_gen/items_<name>.json plus items_arc_full_kr1.json (the 444).

  python scripts/prep_paired_gen_items.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARBITER = "runs/vinf_causal/arbiter_arc_sets_fullsplit.json"
OUTDIR = "runs/paired_gen"
SEED = 0
N_MMLU_PRO = 1000
N_HELLASWAG = 1000
LETTERS = [chr(ord("A") + i) for i in range(26)]


def write(name, items, meta):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, f"items_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"name": name, "n": len(items), **meta, "items": items}, f, indent=2)
    print(f"[write] {p}  n={len(items)}", flush=True)
    return p


def prep_arc():
    """Reuse the arbiter's frozen full split verbatim -- never rebuild it."""
    with open(ARBITER, encoding="utf-8") as f:
        d = json.load(f)
    items = d["items_all"]
    assert len(items) == 1172, f"expected 1172 ARC items, got {len(items)}"
    write("arc_full", items,
          {"source": ARBITER, "note": "verbatim items_all from the full-split arbiter"})

    mc_only = set(d["sets"]["mc_only"])
    assert len(mc_only) == 444, f"expected 444 mc_only, got {len(mc_only)}"
    kr1 = [it for it in items if it["doc_id"] in mc_only]
    assert len(kr1) == 444
    write("arc_full_kr1", kr1,
          {"source": ARBITER, "set": "mc_only",
           "note": "the 444 items the adapter fixes in cold scoring -- KR1 subset"})


def _mc_item(doc_id, question, options, gold_idx, extra=None):
    n = len(options)
    it = {"doc_id": doc_id, "question": question,
          "choices": {"text": [str(o) for o in options], "label": LETTERS[:n]},
          "gold": int(gold_idx)}
    if extra:
        it.update(extra)
    return it


def prep_mmlu_pro():
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_cat = defaultdict(list)
    for i, row in enumerate(ds):
        by_cat[row["category"]].append((i, row))
    total = sum(len(v) for v in by_cat.values())
    print(f"[mmlu-pro] {total} test items across {len(by_cat)} categories", flush=True)

    import random
    rng = random.Random(SEED)
    # proportional allocation, largest-remainder so the total lands exactly on N
    raw = {c: len(v) / total * N_MMLU_PRO for c, v in by_cat.items()}
    take = {c: int(x) for c, x in raw.items()}
    rem = sorted(by_cat, key=lambda c: raw[c] - take[c], reverse=True)
    k = 0
    while sum(take.values()) < N_MMLU_PRO:
        take[rem[k % len(rem)]] += 1
        k += 1

    items = []
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat], key=lambda t: t[0])
        pick = rng.sample(pool, min(take[cat], len(pool)))
        for i, row in sorted(pick, key=lambda t: t[0]):
            items.append(_mc_item(i, row["question"], row["options"],
                                  row["answer_index"], {"category": cat}))
    items.sort(key=lambda it: it["doc_id"])
    print(f"[mmlu-pro] sampled {len(items)}; "
          f"per-category {dict(Counter(it['category'] for it in items))}", flush=True)
    write("mmlu_pro", items,
          {"source": "TIGER-Lab/MMLU-Pro:test", "seed": SEED,
           "sampling": "proportional stratified by category, largest-remainder"})


def prep_hellaswag():
    from datasets import load_dataset
    import random
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(ds)), N_HELLASWAG))
    items = []
    for i in idx:
        row = ds[i]
        ctx = (row["ctx"] or "").strip()
        q = (f"{row['activity_label']}: {ctx}" if row.get("activity_label") else ctx)
        items.append(_mc_item(i, f"Which ending is most plausible?\n{q}",
                              row["endings"], int(row["label"])))
    write("hellaswag", items,
          {"source": "Rowan/hellaswag:validation", "seed": SEED,
           "sampling": f"random {N_HELLASWAG}",
           "note": "cold + letter channels only -- plausibility continuation, not a question"})


def prep_winogrande():
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    items = []
    for i, row in enumerate(ds):
        q = f"Fill in the blank:\n{row['sentence']}"
        gold = 0 if str(row["answer"]).strip() == "1" else 1
        items.append(_mc_item(i, q, [row["option1"], row["option2"]], gold))
    write("winogrande", items,
          {"source": "allenai/winogrande:winogrande_xl:validation",
           "sampling": "full validation split",
           "note": "cold + letter channels only"})


def main():
    prep_arc()
    for fn in (prep_mmlu_pro, prep_hellaswag, prep_winogrande):
        try:
            fn()
        except Exception as e:                     # dataset not cached / offline
            print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
