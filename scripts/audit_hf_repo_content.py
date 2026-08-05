"""Audit a Hugging Face repo for gated-dataset content -- by CONTENT and over its HISTORY.

Why this exists. A HF repo IS a git repo: a file deleted in a later commit is still in the
history, and flipping the repo public exposes every revision. That is exactly how 31 gated
files survived in this project's GitHub repo. Auditing the *current tree* therefore does not
answer the question "is it safe to make this public".

Two phases, both usable alone:

  --history   metadata only (no blob download): union of every path that existed in ANY
              commit, diffed against the current tree. If the union equals the current tree,
              nothing was ever deleted or renamed, and a content audit of the current tree IS
              the audit of the history. That is the cleanest possible verdict and it is
              reported as such.

  --content   download every *text* file (json/jsonl/csv/log/md/txt) and scan it for gated
              content. Binary tensors (.pt/.safetensors) are NOT opened -- that limit is
              declared, not hidden (see DATA-LICENSES.md).

The fingerprints are built from BOTH sides:
  * schema markers   -- the column names gated dumps carry ("Pre-Revision Question", ...)
  * real item text   -- stems lifted at runtime from the gated files still on local disk

The second half matters: a marker sweep for the `Pre-Revision *` columns found 17 files, and
sweeping for the actual item text found 27. The worst GPQA file carried no such column at all.
Schema is not content.

Usage:
  python scripts/audit_hf_repo_content.py --repo sebacascardo87/msap-p9-onaxis-20260709 --history
  python scripts/audit_hf_repo_content.py --repo <id> --history --content --out runs/release_audit/<name>.json
  python scripts/audit_hf_repo_content.py --self-test          # positive control, no network
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------- fingerprints

# Schema markers: column/field names that only gated dumps carry.
SCHEMA_MARKERS = {
    "GPQA": [
        "Pre-Revision Question",
        "Pre-Revision Correct Answer",
        "Pre-Revision Explanation",
        "Explanation_NEV",
        "Self-reported question-writing time",
    ],
    "EWoK": ['"source": "ewok"', '"source":"ewok"', "ewok_domain"],
    "LIMA": ["genlima_", "gen_corpus_lima"],
}

# Local files that still hold the gated material (untracked, gitignored). Used for two
# things: lifting real item text, and as the positive control.
GATED_LOCAL = {
    "GPQA": [
        "runs/vinf_causal/d_mc_gpqa.json",
        "runs/phase3_adapters/factual/eval/gpqa_diamond.json",
    ],
    "EWoK": ["runs/phase3_adapters/kr_opcion_d/train/ewok_pairs.jsonl"],
    "LIMA": ["runs/mvp1_v3/gen_corpus_lima.jsonl"],
}

TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".log", ".md", ".txt", ".yaml", ".yml", ".py", ".sh"}
STEM_LEN = 60          # chars of item text used as a fingerprint
# Lift EVERY prose stem available, not a sample. A cap of 25 is a coverage hole: a log that
# echoes one GPQA item the cap happened to exclude reads as clean. `phase1_bc.log` -- a file
# no name-based sweep would flag -- was caught only because its item happened to be in range.
STEMS_PER_DATASET = 100_000


def _is_prose(s: str) -> bool:
    """Does this string look like natural-language item text (vs a path, hash, or id)?

    The bar matters. A first version of this lifted stems with a regex over the RAW text,
    and `"([^"\\]{60,400})"` happily matched the JSON *structure between* two strings --
    e.g. `": [\\n [\\n -35.0,\\n false"` from `"resps": [[-35.0, false]]`. Those fragments
    occur in every lm-eval dump, so the scanner flagged eight clean files. Parse, then judge
    the value; never regex the container.
    """
    if len(s) < STEM_LEN or s.count(" ") < 6:
        return False
    if s.startswith(("runs/", "http", "/", "__", "hf://")):
        return False
    letters = sum(ch.isalpha() for ch in s)
    return letters / len(s) > 0.65


def _walk_strings(obj, out: list[str], budget: int = 20000) -> None:
    if len(out) >= budget:
        return
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_strings(v, out, budget)
    elif isinstance(obj, list):
        for v in obj:
            _walk_strings(v, out, budget)


def _lift_stems(path: Path, n: int) -> list[str]:
    """Pull distinct chunks of REAL item text out of a gated file, by PARSING it."""
    values: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    try:
        _walk_strings(json.loads(raw), values)
    except json.JSONDecodeError:                       # .jsonl
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                _walk_strings(json.loads(line), values)
            except json.JSONDecodeError:
                continue

    stems: list[str] = []
    for s in values:
        if not _is_prose(s):
            continue
        stem = s[:STEM_LEN].strip()
        if stem and stem not in stems:
            stems.append(stem)
        if len(stems) >= n:
            break
    return stems


def build_fingerprints(root: Path) -> dict[str, list[str]]:
    fp: dict[str, list[str]] = {}
    for ds, markers in SCHEMA_MARKERS.items():
        fp[ds] = list(markers)
    for ds, rels in GATED_LOCAL.items():
        for rel in rels:
            p = root / rel
            if p.exists():
                lifted = _lift_stems(p, STEMS_PER_DATASET)
                fp[ds].extend(lifted)
                print(f"  [fp] {ds}: +{len(lifted)} item stems from {rel}", flush=True)
            else:
                print(f"  [fp] {ds}: {rel} NOT on disk -- schema markers only", flush=True)
    return fp


def scan_text(text: str, fingerprints: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return {dataset: [matched fingerprints]} for a blob of text."""
    hits: dict[str, list[str]] = {}
    for ds, pats in fingerprints.items():
        matched = [p for p in pats if p in text]
        if matched:
            hits[ds] = matched
    return hits


# ---------------------------------------------------------------- history phase

def audit_history(api, repo_id: str, repo_type: str) -> dict:
    from huggingface_hub import list_repo_tree

    commits = list(api.list_repo_commits(repo_id=repo_id, repo_type=repo_type))
    print(f"  [history] {len(commits)} commits", flush=True)

    current = set()
    for e in list_repo_tree(repo_id, repo_type=repo_type, recursive=True):
        if getattr(e, "size", None) is not None:      # RepoFile, not RepoFolder
            current.add(e.path)

    union: set[str] = set(current)
    per_commit = []
    # (path, blob_id) -> a revision where that blob is reachable. Distinct blobs are what
    # must be scanned: a path that survived every commit can still have carried different
    # CONTENT in an earlier revision, and that content is still published when the repo is.
    blobs: dict[tuple[str, str], str] = {}
    for i, c in enumerate(commits, 1):
        try:
            entries = [e for e in list_repo_tree(repo_id, repo_type=repo_type,
                                                 revision=c.commit_id, recursive=True)
                       if getattr(e, "size", None) is not None]
        except Exception as exc:                      # revision unreachable -> declare it
            print(f"  [{i}/{len(commits)}] {c.commit_id[:8]} UNREADABLE: {exc}", flush=True)
            per_commit.append({"sha": c.commit_id, "error": str(exc)})
            continue
        paths = {e.path for e in entries}
        for e in entries:
            key = (e.path, getattr(e, "blob_id", "") or "")
            blobs.setdefault(key, c.commit_id)
        union |= paths
        per_commit.append({"sha": c.commit_id, "title": c.title, "n_files": len(paths)})
        print(f"  [{i}/{len(commits)}] {c.commit_id[:8]} {len(paths):4d} files  "
              f"union={len(union)} distinct_blobs={len(blobs)}", flush=True)

    vanished = sorted(union - current)
    n_paths_with_multiple_blobs = len({p for p, _ in blobs}) and sum(
        1 for p in {p for p, _ in blobs} if sum(1 for q, _ in blobs if q == p) > 1)
    return {
        "n_commits": len(commits),
        "n_current": len(current),
        "n_union": len(union),
        "current_tree": sorted(current),
        "paths_only_in_history": vanished,
        "history_equals_current": not vanished,
        "n_distinct_blobs": len(blobs),
        "n_paths_with_multiple_blobs": n_paths_with_multiple_blobs,
        "blobs": [{"path": p, "blob_id": b, "revision": rev} for (p, b), rev in blobs.items()],
        "per_commit": per_commit,
    }


# ---------------------------------------------------------------- content phase

def audit_content(api, repo_id: str, repo_type: str, targets: list[tuple[str, str | None]],
                  fingerprints: dict[str, list[str]]) -> dict:
    """Scan (path, revision) pairs. Pass every DISTINCT blob from the history, not just HEAD."""
    from huggingface_hub import hf_hub_download

    findings, skipped, scanned = {}, [], 0
    text = [(p, r) for p, r in targets if Path(p).suffix.lower() in TEXT_SUFFIXES]
    binary = sorted({p for p, _ in targets if Path(p).suffix.lower() not in TEXT_SUFFIXES})
    print(f"  [content] {len(text)} text blobs to scan, "
          f"{len(binary)} binary paths NOT opened (declared)", flush=True)

    for i, (path, revision) in enumerate(text, 1):
        label = path if revision is None else f"{path}@{revision[:8]}"
        try:
            local = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=path,
                                    revision=revision)
            body = Path(local).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            skipped.append({"path": path, "revision": revision, "error": str(exc)})
            print(f"  [{i}/{len(text)}] SKIP {label}: {exc}", flush=True)
            continue
        scanned += 1
        hits = scan_text(body, fingerprints)
        print(f"  [{i}/{len(text)}] {label} ({len(body):,} chars)"
              f"{'  <-- HIT' if hits else ''}", flush=True)
        if hits:
            findings[label] = {ds: pats[:3] for ds, pats in hits.items()}

    return {
        "n_text_blobs_scanned": scanned,
        "n_binary_paths_not_opened": len(binary),
        "binary_not_opened": binary,
        "skipped": skipped,
        "findings": findings,
        "clean": not findings,
    }


# ---------------------------------------------------------------- positive control

def self_test(root: Path) -> int:
    """An auditor that cannot fail does not measure. Fire the scanner at known-dirty files."""
    print("== POSITIVE CONTROL ==", flush=True)
    fp = build_fingerprints(root)
    ok = True
    for ds, rels in GATED_LOCAL.items():
        for rel in rels:
            p = root / rel
            if not p.exists():
                print(f"  SKIP (absent): {rel}", flush=True)
                continue
            hits = scan_text(p.read_text(encoding="utf-8", errors="replace"), fp)
            fired = ds in hits
            print(f"  {'FIRES  ' if fired else 'SILENT '} {rel}  -> {sorted(hits)}", flush=True)
            ok &= fired
    # Negative controls: files verified clean must NOT fire. The last one is a full lm-eval
    # per-item dump of NON-gated tasks -- the exact file class that the first version of the
    # stem-lifter false-positived on, so it stays in the control set permanently.
    for rel in ["runs/vinf_causal/arbiter_gpqa_cot.json",
                "runs/vinf_causal/lift_difficulty_stratification_leaderboard_gpqa_diamond.json",
                "runs/eval_bulletproof/R1_v7mc.json"]:
        p = root / rel
        if p.exists():
            hits = scan_text(p.read_text(encoding="utf-8", errors="replace"), fp)
            print(f"  {'FALSE+ ' if hits else 'quiet  '} {rel}  -> {sorted(hits)}", flush=True)
            ok &= not hits
    print(f"\n  SELF-TEST: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--content", action="store_true")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scan-local", type=Path,
                    help="Scan a STAGED directory before uploading it. This is the audit that "
                         "matters for a new repo: the bytes that will actually travel.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test(args.root))

    if args.scan_local:
        fingerprints = build_fingerprints(args.root)
        files = sorted(p for p in args.scan_local.rglob("*") if p.is_file())
        text = [p for p in files if p.suffix.lower() in TEXT_SUFFIXES]
        binary = [p for p in files if p.suffix.lower() not in TEXT_SUFFIXES]
        print(f"  [staged] {len(files)} files: {len(text)} text scanned, "
              f"{len(binary)} binary NOT opened (declared)", flush=True)
        findings = {}
        for i, p in enumerate(text, 1):
            hits = scan_text(p.read_text(encoding="utf-8", errors="replace"), fingerprints)
            rel = p.relative_to(args.scan_local).as_posix()
            print(f"  [{i}/{len(text)}] {rel}{'  <-- HIT' if hits else ''}", flush=True)
            if hits:
                findings[rel] = {ds: pats[:3] for ds, pats in hits.items()}
        rep = {"staged_dir": str(args.scan_local), "n_files": len(files),
               "n_text_scanned": len(text),
               "binary_not_opened": [p.relative_to(args.scan_local).as_posix() for p in binary],
               "findings": findings, "clean": not findings}
        print(f"\n  STAGED: {'CLEAN' if rep['clean'] else 'HITS: ' + ', '.join(findings)}",
              flush=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            print(f"  saved: {args.out}", flush=True)
        sys.exit(0 if rep["clean"] else 1)

    if not args.repo:
        sys.exit("[ABORT] --repo is required unless --self-test")

    from huggingface_hub import HfApi
    api = HfApi()

    print(f"== {args.repo} ({args.repo_type}) ==", flush=True)
    fingerprints = build_fingerprints(args.root)
    report = {"repo": args.repo, "repo_type": args.repo_type,
              "n_fingerprints": {k: len(v) for k, v in fingerprints.items()}}

    targets = None
    if args.history:
        report["history"] = audit_history(api, args.repo, args.repo_type)
        h = report["history"]
        # Every distinct blob over the whole history -- that is what publishing exposes.
        targets = [(b["path"], b["revision"]) for b in h["blobs"]]
        print(f"\n  HISTORY: {h['n_commits']} commits, current={h['n_current']}, "
              f"union={h['n_union']}, only-in-history={len(h['paths_only_in_history'])}, "
              f"distinct_blobs={h['n_distinct_blobs']} "
              f"({h['n_paths_with_multiple_blobs']} paths changed content)", flush=True)
        for p in h["paths_only_in_history"]:
            print(f"    ONLY-IN-HISTORY: {p}", flush=True)

    if args.content:
        if targets is None:
            from huggingface_hub import list_repo_tree
            targets = [(e.path, None) for e in list_repo_tree(args.repo, repo_type=args.repo_type,
                                                              recursive=True)
                       if getattr(e, "size", None) is not None]
        report["content"] = audit_content(api, args.repo, args.repo_type, targets, fingerprints)
        c = report["content"]
        print(f"\n  CONTENT: scanned {c['n_text_blobs_scanned']} text blobs, "
              f"{c['n_binary_paths_not_opened']} binary paths not opened, "
              f"{'CLEAN' if c['clean'] else 'HITS: ' + ', '.join(c['findings'])}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
