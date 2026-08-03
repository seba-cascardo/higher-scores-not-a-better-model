#!/usr/bin/env python3
r"""Resolve, stamp and recover WHICH model snapshot an eval artifact actually ran on.

Why this exists (B64, 2026-08-03)
---------------------------------
The paper declares a single checkpoint pin, `google/gemma-4-31B-it` at revision
`842da3794eaa...`. Two independent pre-release audits went looking for that SHA
in the result artifacts, found `"model_sha": ""` in every one of them, and both
concluded the same thing: the runs cannot be pinned to a snapshot post hoc.

That conclusion was wrong, and it was wrong in the project's own signature way --
`an artifact is identified by its CONTENT, not by the name of the field`. lm-eval
does leave `model_sha` empty, but it records `config.model_args` verbatim, and on
the pod the base model is passed as a Hugging Face cache path:

    base=/workspace/hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5.../

So the snapshot was in the artifacts the whole time, one field over from where
everybody looked. `recover_snapshots()` below extracts it, and it resolves the
question completely: the June headline campaign and the causal spine ran on
`3548...` (the OLD snapshot), while the July full-split replication ran on
`842da...` (the NEW one). The two differ in `chat_template.jinja` and
`tokenizer_config.json` only -- `config.json` and `tokenizer.json` are
byte-identical -- so no weight changed, but this paper is protocol-bound to the
chat template and the new template's own header advertises "Fixed tool-calling
loops, turn closures".

Going forward, `stamp()` closes the class of finding rather than re-deriving it:
call it where a harness writes its result JSON and the snapshot plus the hash of
the chat template that was actually applied land in the artifact.

Usage:
    # derive the map from what is already on disk
    python scripts/model_provenance.py --recover runs outputs

    # inside a harness, before writing results
    from scripts.model_provenance import stamp
    payload["model_provenance"] = stamp(model_path_or_repo, tokenizer)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SNAPSHOT_RE = re.compile(r"snapshots[/\\]([0-9a-f]{40})")
LMEVAL_DATE_RE = re.compile(r'"date"\s*:\s*([0-9.]+)')

# The two snapshots this project has actually run on, and what separates them.
KNOWN_SNAPSHOTS = {
    "3548789868c5356dbf307c98e6f609007b82b3eb": "old (pre-2026-07-09 chat template)",
    "842da3794eaa0b77d5f08bae87a17459d91ff475": "new (2026-07-09 chat template; = main since 2026-07-20)",
}


# --------------------------------------------------------------------------- #
# Forward direction: stamp provenance into new artifacts.
# --------------------------------------------------------------------------- #
def chat_template_hash(tokenizer=None, model_dir: str | Path | None = None) -> str | None:
    """SHA-256 of the chat template actually in force, or None if there is none.

    Prefers the live tokenizer object -- that is the template that will be
    applied -- and falls back to `chat_template.jinja` next to the weights.
    """
    text = None
    if tokenizer is not None:
        text = getattr(tokenizer, "chat_template", None)
    if text is None and model_dir is not None:
        candidate = Path(model_dir) / "chat_template.jinja"
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="replace")
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_snapshot(model_path_or_repo: str | Path) -> str | None:
    """The 40-hex snapshot for a model, from its cache path or from the Hub."""
    m = SNAPSHOT_RE.search(str(model_path_or_repo))
    if m:
        return m.group(1)
    # A local directory that is a symlink into the cache still resolves.
    p = Path(model_path_or_repo)
    if p.exists():
        m = SNAPSHOT_RE.search(str(p.resolve()))
        if m:
            return m.group(1)
    try:  # a bare repo id: ask the Hub what the revision currently resolves to
        from huggingface_hub import HfApi

        return HfApi().model_info(str(model_path_or_repo)).sha
    except Exception:
        return None


def stamp(model_path_or_repo: str | Path, tokenizer=None) -> dict:
    """The provenance block to embed in a result artifact.

    `revision: "main"` is not a pin -- upstream moved on 2026-07-20 -- so the
    resolved snapshot and the template hash are the two fields that make a run
    reproducible. Neither is expensive; both were missing.
    """
    snap = resolve_snapshot(model_path_or_repo)
    return {
        "model": str(model_path_or_repo),
        "resolved_snapshot": snap,
        "snapshot_note": KNOWN_SNAPSHOTS.get(snap or "", "unrecognised snapshot"),
        "chat_template_sha256": chat_template_hash(tokenizer, model_path_or_repo),
        "stamped_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Backward direction: recover the map from artifacts that predate the stamp.
# --------------------------------------------------------------------------- #
def recover_snapshots(roots: list[str]) -> dict:
    """Scan artifacts for embedded snapshot paths -> {file: {snapshots, date}}."""
    found: dict[str, dict] = {}
    for root in roots:
        base = REPO / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in {".json", ".log", ".txt", ".md"} or not p.is_file():
                continue
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            snaps = sorted(set(SNAPSHOT_RE.findall(raw)))
            if not snaps:
                continue
            date = None
            if (m := LMEVAL_DATE_RE.search(raw)) is not None:
                try:
                    date = (
                        datetime.datetime.fromtimestamp(float(m.group(1)), datetime.UTC)
                        .date()
                        .isoformat()
                    )
                except Exception:
                    pass
            found[p.relative_to(REPO).as_posix()] = {"snapshots": snaps, "date": date}
    return found


def summarise(found: dict) -> dict:
    """Group per campaign directory, so the map is readable at paper scale."""
    campaigns: dict[str, dict] = {}
    for rel, info in found.items():
        parts = rel.split("/")
        key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
        row = campaigns.setdefault(key, {"snapshots": set(), "dates": set(), "files": 0})
        row["snapshots"].update(info["snapshots"])
        if info["date"]:
            row["dates"].add(info["date"])
        row["files"] += 1
    return {
        k: {
            "snapshots": sorted(v["snapshots"]),
            "date_span": (
                [min(v["dates"]), max(v["dates"])] if v["dates"] else None
            ),
            "files": v["files"],
        }
        for k, v in sorted(campaigns.items())
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recover", nargs="*", default=["runs", "outputs"], metavar="ROOT")
    ap.add_argument("--out", default="runs/provenance/model_snapshot_map.json")
    args = ap.parse_args()

    found = recover_snapshots(args.recover)
    summary = summarise(found)

    print(f"artifacts carrying a snapshot : {len(found)}")
    print(f"campaigns                     : {len(summary)}\n")
    print(f"{'campaign':<34} {'snapshot(s)':<26} {'dates':<24} files")
    for camp, info in summary.items():
        tags = "+".join(
            ("OLD-3548" if s.startswith("3548") else "NEW-842d" if s.startswith("842d") else s[:8])
            for s in info["snapshots"]
        )
        span = "-" if not info["date_span"] else (
            info["date_span"][0]
            if info["date_span"][0] == info["date_span"][1]
            else f"{info['date_span'][0]}..{info['date_span'][1]}"
        )
        print(f"{camp:<34} {tags:<26} {span:<24} {info['files']}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                "known_snapshots": KNOWN_SNAPSHOTS,
                "note": (
                    "lm-eval leaves model_sha empty but records the cache path in "
                    "config.model_args; that path carries the snapshot. Recovered "
                    "post hoc -- see the module docstring."
                ),
                "by_campaign": summary,
                "by_file": found,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
