"""OQ1 functional axis-angle analysis (E2) — process-LoFiT GATE. LOCAL, CPU, €0.

Reads the .pt produced by ``scripts/extract_functional_axis_perhead.py``
(schema_version 'oq1-func-1.0') and answers the §5 question of the axis paper
directly: are the *functional* directions ``v_inf[h]`` (discriminates correct-vs-
wrong in INFERENCE-form) and ``v_ret[h]`` (idem in RETRIEVAL-form) orthogonal /
separable, or do they share a direction? This is the gate of process-LoFiT (an
orthogonal-decomposition loss only makes sense if the directions are separable by
projection).

Pre-registration that this script serves (DO NOT re-derive the verdict map here):
  docs/protocols/2026-06-22-oq1-vg-execution-prereg.md  §2 (E2 functional)

VERDICT MAP (pre-registered §2 — crossed on the *whitened* mean cos over eligible heads):
  - whitened mean WITHIN the null (BCa CI overlaps the null band ≈ 0, neither side excluded)
        => ORTHOGONAL  => §5 survives; disentanglement-by-projection viable
        => process-LoFiT GATE = GREEN.
  - whitened mean >> 0 (BCa lower bound > 0, above the null band)
        => ALIGNED     => §5 false / puzzle (sharpening one improves the other)
        => REWRITE §5; process-LoFiT by orthogonal-decomp dubious.
  - whitened mean << 0 (BCa upper bound < 0, below the null band)
        => ANTI        => single functional axis => disentanglement impossible
        => process-LoFiT GATE = RED (kill before spending corpus).

REUSE (imported, NOT re-implemented) from ``scripts.probe_v7_lofit_head_selection``:
  - fit_massmean(H_c, H_w, scale=False) -> direction (mu_c - mu_w). Used here ONLY
    to RECOMPUTE v_inf/v_ret from the stored per-class means as a cheap integrity
    cross-check vs the extractor's saved v_inf/v_ret (mass-mean dir == mu_c - mu_w).
  - park_whiten is the canonical whitening; we reuse its EXACT math (pooled cov,
    eigh, eigvals clamped at 1e-6) but operate on the per-head covariance the
    extractor saved (we whiten DIRECTIONS, not sample matrices).

INPUT SCHEMA (what extract_functional_axis_perhead.py emits, 'oq1-func-1.0'):
  v_inf, v_ret              : Tensor[n_lay, num_q_heads, head_dim]  (massmean dirs)
  mu_inf_correct/wrong      : Tensor[n_lay, num_q_heads, head_dim]
  mu_ret_correct/wrong      : Tensor[n_lay, num_q_heads, head_dim]
  cov_inf, cov_ret          : dict[(layer,head) -> Tensor[head_dim,head_dim]]  (ELIGIBLE heads only)
  acc_inf, acc_ret          : Tensor[n_lay, num_q_heads]
  eligible_cells_layer_head : list[[layer,head]]
  theta_mc                  : Tensor[n_lay, num_q_heads, head_dim]  (alpha*theta, applied)
  captured_indices, num_q_heads, head_dim, ...

COVARIANCE NOTE (the extractor stores cov PER FAMILY, eligible heads only):
  The pre-reg names a single per-head base covariance for Mahalanobis whitening,
  but the extractor (correctly, the families differ) stores cov_inf AND cov_ret.
  We whiten in the COMMON metric W = (0.5*(cov_inf+cov_ret))^{-1/2} (the per-head
  base-activation metric averaged over both families), and additionally report the
  per-family-whitened cos as a SENSITIVITY check. Conditioning (clamped eigvals) is
  reported. If cov is absent for the eligible heads (e.g. extractor ran with 0
  eligible heads), whitened cos is impossible -> LOUD warning + raw cos only.

Usage (CPU, post-extraction):
  python scripts/probe_oq1_functional_angle.py \\
      --acts runs/oq1_functional_axis/functional_directions.pt \\
      --acc-threshold 0.60 \\
      --out runs/oq1_functional_axis/oq1_functional_angle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os

import numpy as np
import torch

# Make `from scripts.X import ...` resolve whether invoked as `python scripts/foo.py`
# (script dir on path, repo root NOT) or `python -m scripts.foo` (repo root on path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the canonical probe-direction + whitening math. Importing (not
# re-implementing) keeps geometry byte-identical to V7 head selection.
from scripts.probe_v7_lofit_head_selection import fit_massmean, park_whiten  # noqa: F401

# This is a LOCAL CPU script that may run under a Windows console (cp1252). The
# status messages contain non-ASCII (section signs, ~, sqrt, currency). Force
# UTF-8 with replacement so a console-encoding mismatch never crashes the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-reconfigurable stream (e.g. pipe)
        pass


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _whitening_from_cov(cov: torch.Tensor, eps: float = 1e-6):
    """Mahalanobis whitening transform W = cov^{-1/2} from a per-head covariance.

    Mirrors scripts.probe_v7_lofit_head_selection.park_whiten EXACTLY: eigh of the
    (already pooled, already mean-centered by the extractor) covariance, eigvals
    clamped at 1e-6, W = V diag(eig^{-1/2}) V^T. Returns (W, eigvals_raw).
    cos in whitened space = cos(W v_inf, W v_ret) = the Mahalanobis-normalized
    inner product v_inf^T cov^{-1} v_ret.
    """
    cov = cov.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals_clamped = eigvals.clamp(min=eps)
    W = eigvecs @ torch.diag(eigvals_clamped.rsqrt()) @ eigvecs.T
    return W, eigvals


def _whiten_vec(W: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return W @ v.to(W.dtype)


def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float((a @ b) / (na * nb))


def _grid_get(grid: torch.Tensor, arr_idx: int, hi: int) -> torch.Tensor:
    return grid[arr_idx, hi, :]


# ---------------------------------------------------------------------------
# Build per-head records from the 'oq1-func-1.0' schema
# ---------------------------------------------------------------------------
def _heads_from_oq1_schema(payload: dict, acc_threshold: float):
    """Build per-head records from the extractor's grid schema.

    Eligibility here is recomputed from the saved acc grids at THIS --acc-threshold
    (which may be stricter than the extractor's --elig-acc). Whitened cos requires
    the per-head covariance, which the extractor stored only for heads eligible at
    ITS threshold. So a head can be eligible-here but lack covariance if this
    threshold is < the extractor's; such heads get raw cos only (flagged).
    """
    v_inf = payload["v_inf"]
    v_ret = payload["v_ret"]
    acc_inf = payload["acc_inf"]
    acc_ret = payload["acc_ret"]
    captured = [int(x) for x in payload["captured_indices"]]
    cov_inf = payload.get("cov_inf", {}) or {}
    cov_ret = payload.get("cov_ret", {}) or {}
    theta_mc = payload.get("theta_mc")          # grid [n_lay, nH, D] or None
    mu_inf_c = payload.get("mu_inf_correct")
    mu_inf_w = payload.get("mu_inf_wrong")
    mu_ret_c = payload.get("mu_ret_correct")
    mu_ret_w = payload.get("mu_ret_wrong")

    n_lay = v_inf.shape[0]
    n_heads = v_inf.shape[1]

    # cov dicts are keyed by (layer, head) tuples (the canonical join key).
    def _cov_lookup(d, li, hi):
        return d.get((li, hi), d.get((int(li), int(hi))))

    records = []
    any_cov = False
    total = n_lay * n_heads
    done = 0
    for arr_idx in range(n_lay):
        li = captured[arr_idx]
        for hi in range(n_heads):
            done += 1
            a_inf = float(acc_inf[arr_idx, hi])
            a_ret = float(acc_ret[arr_idx, hi])
            eligible = (a_inf >= acc_threshold) and (a_ret >= acc_threshold)
            rec = {"layer": li, "head": hi, "acc_inf": a_inf, "acc_ret": a_ret,
                   "eligible": bool(eligible)}
            if eligible:
                vi = _grid_get(v_inf, arr_idx, hi)
                vr = _grid_get(v_ret, arr_idx, hi)
                rec["cos_raw"] = _cos(vi, vr)

                # Integrity cross-check: massmean(mu_c, mu_w) == mu_c - mu_w == v.
                # Confirms the saved direction equals what fit_massmean would give.
                if mu_inf_c is not None and mu_inf_w is not None:
                    vi_recompute = fit_massmean(
                        _grid_get(mu_inf_c, arr_idx, hi).unsqueeze(0),
                        _grid_get(mu_inf_w, arr_idx, hi).unsqueeze(0), scale=False)
                    rec["v_inf_recompute_maxabs_err"] = float(
                        (vi_recompute - vi).abs().max())

                ci = _cov_lookup(cov_inf, li, hi)
                cr = _cov_lookup(cov_ret, li, hi)
                if ci is not None and cr is not None:
                    any_cov = True
                    # Common-metric whitening: average the two family covariances.
                    cov_common = 0.5 * (ci.to(torch.float64) + cr.to(torch.float64))
                    W, eig = _whitening_from_cov(cov_common)
                    rec["cos_whitened"] = _cos(_whiten_vec(W, vi), _whiten_vec(W, vr))
                    rec["n_near_zero_eig"] = int((eig < 1e-6).sum())
                    rec["min_eig"] = float(eig.min())
                    rec["max_eig"] = float(eig.max())
                    rec["cond_number"] = float(eig.max() / eig.clamp(min=1e-12).min())
                    # Per-family sensitivity.
                    Wi, _ = _whitening_from_cov(ci)
                    Wr, _ = _whitening_from_cov(cr)
                    rec["cos_whitened_inf_metric"] = _cos(_whiten_vec(Wi, vi), _whiten_vec(Wi, vr))
                    rec["cos_whitened_ret_metric"] = _cos(_whiten_vec(Wr, vi), _whiten_vec(Wr, vr))
                    # theta_mc cross-check (§134 bridge), whitened in common metric.
                    if theta_mc is not None:
                        tmc = _grid_get(theta_mc, arr_idx, hi)
                        if float(torch.linalg.norm(tmc)) > 1e-12:
                            rec["cos_inf_thetamc_raw"] = _cos(vi, tmc)
                            rec["cos_inf_thetamc_whitened"] = _cos(
                                _whiten_vec(W, vi), _whiten_vec(W, tmc))
                else:
                    rec["cov_missing"] = True
                    # Still allow a raw theta_mc cross-check without covariance.
                    if theta_mc is not None:
                        tmc = _grid_get(theta_mc, arr_idx, hi)
                        if float(torch.linalg.norm(tmc)) > 1e-12:
                            rec["cos_inf_thetamc_raw"] = _cos(vi, tmc)
            records.append(rec)
            if done % 500 == 0:
                print(f"  [{done}/{total}] heads processed", flush=True)
    return records, any_cov


# ---------------------------------------------------------------------------
# Null + BCa bootstrap over heads
# ---------------------------------------------------------------------------
def random_unit_null_cos(head_dim: int, n_samples: int, seed: int) -> np.ndarray:
    """Random-unit-vector null in dim=head_dim.

    Two iid unit vectors in R^D have cos concentrating at 0 with sd ≈ 1/sqrt(D)
    (concentration of measure). Returns n_samples cos values used to characterize
    the null band that the WHITENED mean must fall inside to read ORTHOGONAL.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_samples, head_dim))
    b = rng.standard_normal((n_samples, head_dim))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def bca_ci(values: np.ndarray, n_boot: int, seed: int,
           alpha: float = 0.05) -> tuple[float, float, float]:
    """BCa (bias-corrected & accelerated) bootstrap CI for the MEAN over heads.

    Returns (lo, mean, hi) at the (1-alpha) level. BCa corrects bias + skew of the
    head-level cos distribution (small n, non-normal). Falls back to percentile if
    the bias/acceleration terms degenerate (n<3 or zero-variance).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    theta_hat = float(np.mean(values)) if n else float("nan")
    if n < 3:
        return (theta_hat, theta_hat, theta_hat)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(values[idx])

    from scipy.stats import norm  # lazy import; scipy is a project dep

    prop_less = float(np.mean(boot < theta_hat))
    prop_less = min(max(prop_less, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = norm.ppf(prop_less)

    # Acceleration via jackknife of the mean.
    jack = np.array([np.mean(np.delete(values, j)) for j in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def _adjust(z):
        denom = 1 - a_hat * (z0 + z)
        if denom == 0:
            return np.nan
        return norm.cdf(z0 + (z0 + z) / denom)

    p_lo, p_hi = _adjust(z_lo), _adjust(z_hi)
    if not (np.isfinite(p_lo) and np.isfinite(p_hi)):
        lo = float(np.percentile(boot, 100 * (alpha / 2)))
        hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return (lo, theta_hat, hi)
    lo = float(np.percentile(boot, 100 * p_lo))
    hi = float(np.percentile(boot, 100 * p_hi))
    return (lo, theta_hat, hi)


# ---------------------------------------------------------------------------
# Verdict (pre-registered §2)
# ---------------------------------------------------------------------------
def decide_verdict(metric_name: str, ci_lo: float, mean: float, ci_hi: float,
                   null_lo: float, null_hi: float) -> tuple[str, str, str]:
    """Pre-registered §2 verdict map on the WHITENED mean cos over eligible heads.

    "within the null" => the metric's BCa CI overlaps the null band and does not
    exclude 0 on the ALIGNED/ANTI side. Concretely:
      - BCa LB > 0 AND LB above the null band -> ALIGNED -> rewrite §5 (gate dubious)
      - BCa UB < 0 AND UB below the null band -> ANTI    -> process-LoFiT GATE = RED
      - otherwise (CI straddles/overlaps the null band) -> ORTHOGONAL -> GATE GREEN
    Returns (verdict, gate, rationale).
    """
    if ci_lo > 0 and ci_lo > null_hi:
        return ("ALIGNED",
                "DUBIOUS_REWRITE_SECTION_5",
                f"{metric_name} BCa LB {ci_lo:+.4f} > 0 and above null band "
                f"[{null_lo:+.4f},{null_hi:+.4f}]: sharpening inference improves "
                f"retrieval — contradicts the §5 spillover gloss. Rewrite §5; "
                f"process-LoFiT orthogonal-decomp dubious.")
    if ci_hi < 0 and ci_hi < null_lo:
        return ("ANTI",
                "RED",
                f"{metric_name} BCa UB {ci_hi:+.4f} < 0 and below null band "
                f"[{null_lo:+.4f},{null_hi:+.4f}]: single functional axis "
                f"(anti-collinear) — disentanglement impossible by construction. "
                f"process-LoFiT GATE = RED.")
    return ("ORTHOGONAL",
            "GREEN",
            f"{metric_name} mean {mean:+.4f} BCa CI [{ci_lo:+.4f},{ci_hi:+.4f}] "
            f"within / overlapping the null band [{null_lo:+.4f},{null_hi:+.4f}]: "
            f"functional inference/retrieval directions orthogonal. §5 survives; "
            f"disentanglement-by-projection viable. process-LoFiT GATE = GREEN.")


def _summarize(values) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True,
                    help="Path to the .pt from extract_functional_axis_perhead.py")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--acc-threshold", type=float, default=0.60,
                    help="Per-head probe acc required in BOTH families to be eligible "
                         "(pre-reg §2: 'señal en AMBAS familias'). NOTE: if set BELOW "
                         "the extractor's --elig-acc, covariance is missing for the "
                         "extra heads (whitened cos uses only heads with stored cov).")
    ap.add_argument("--n-boot", type=int, default=10000, help="BCa bootstrap resamples")
    ap.add_argument("--n-null", type=int, default=200000,
                    help="Random-unit null samples for the orthogonality band")
    ap.add_argument("--min-elig-gate", type=int, default=10,
                    help="Min eligible heads (with whitened cos) to let the GATE fire "
                         "GREEN/RED with confidence. With too few heads the null band "
                         "widens and biases toward ORTHOGONAL/GREEN — a spurious green "
                         "would greenlight an expensive process-LoFiT corpus. Below this, "
                         "the gate is HELD_LOW_N regardless of the verdict.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("OQ1 functional axis-angle (E2) — process-LoFiT GATE")
    print("=" * 64)
    print(f"  acts: {args.acts}")
    print(f"  acc-threshold (both families): {args.acc_threshold}")
    print(flush=True)

    payload = torch.load(args.acts, weights_only=False, map_location="cpu")
    schema = payload.get("schema_version", "<none>")
    print(f"  schema_version: {schema}")
    print(f"  base: {payload.get('base')}   chat_template: {payload.get('chat_template')}")
    print(f"  retrieval used: {payload.get('retrieval_datasets_used')} "
          f"(fallback_fired={payload.get('retrieval_fallback_fired')})")
    if payload.get("retrieval_fallback_fired"):
        print("  *** retrieval fell back to TruthfulQA — angle is a SENSITIVITY-CHECK, "
              "not primary (pre-reg §2 caveat). ***")

    if "v_inf" not in payload or "v_ret" not in payload:
        print("FATAL: .pt lacks v_inf/v_ret — not an 'oq1-func-1.0' extractor output.",
              file=sys.stderr)
        sys.exit(2)

    head_dim = int(payload.get("head_dim", payload["v_inf"].shape[-1]))
    extractor_elig_acc = payload.get("elig_acc")
    if extractor_elig_acc is not None and args.acc_threshold < float(extractor_elig_acc):
        print(f"  NOTE: --acc-threshold {args.acc_threshold} < extractor elig_acc "
              f"{extractor_elig_acc}: heads eligible here but below the extractor "
              f"threshold have NO stored covariance (raw cos only for those).")

    records, cov_available = _heads_from_oq1_schema(payload, args.acc_threshold)

    elig = [r for r in records if r.get("eligible")]
    n_elig = len(elig)
    print(f"\n  eligible heads (acc>={args.acc_threshold} in BOTH families): "
          f"{n_elig} / {len(records)}", flush=True)

    if n_elig == 0:
        print("\n  *** NO eligible heads — cannot estimate the functional angle. ***")
        print("  Lower --acc-threshold or inspect the extractor's acc_inf/acc_ret grids.")
        result = {
            "acts": str(args.acts), "schema_version": schema,
            "acc_threshold": args.acc_threshold,
            "n_heads_total": len(records), "n_eligible": 0,
            "verdict": "INCONCLUSIVE_NO_ELIGIBLE_HEADS",
            "process_lofit_gate": "HELD_NO_ELIGIBLE_HEADS",
            "records": records,
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n  wrote {out_path}")
        print(json.dumps({"path": str(out_path), "verdict": result["verdict"],
                          "gate": result["process_lofit_gate"], "n_eligible": 0}, indent=2))
        return

    # --- covariance dependency check (LOUD if whitened impossible) -----------
    elig_with_w = [r for r in elig if "cos_whitened" in r]
    has_whitened = len(elig_with_w) > 0 and len(elig_with_w) == n_elig
    n_cov_missing = sum(1 for r in elig if r.get("cov_missing"))
    if not has_whitened:
        print("\n" + "!" * 72)
        print("!! WHITENED COS UNAVAILABLE for some/all eligible heads "
              f"({n_cov_missing}/{n_elig} missing covariance).")
        print("!! The PRE-REG (§2) makes the WHITENED mean the PRIMARY decision metric;")
        print("!! raw cos lives in the non-orthonormal o_proj-input basis and CANNOT")
        print("!! drive the GATE verdict on its own.")
        if n_cov_missing == n_elig:
            print("!! CAUSE: NO covariance stored for ANY eligible head.")
            print("!! FIX: re-run extract_functional_axis_perhead.py with --elig-acc <=")
            print(f"!!      {args.acc_threshold} so it stores pooled cov for these heads,")
            print("!!      OR raise this --acc-threshold to the extractor's elig_acc.")
        else:
            print("!! CAUSE: this --acc-threshold is below the extractor's --elig-acc,")
            print("!!        so the extra heads lack stored covariance.")
            print("!! FIX: set --acc-threshold to the extractor's elig_acc, or re-extract.")
        print("!! Reporting WHITENED on the covariance-available subset + RAW as "
              "sensitivity; GATE is HELD unless all eligible heads have whitened cos.")
        print("!" * 72 + "\n", flush=True)

    # --- aggregate metrics ---------------------------------------------------
    cos_raw_vals = [r["cos_raw"] for r in elig if np.isfinite(r.get("cos_raw", np.nan))]
    summary_raw = _summarize(cos_raw_vals)
    cos_w_vals = [r["cos_whitened"] for r in elig if np.isfinite(r.get("cos_whitened", np.nan))]
    summary_whitened = _summarize(cos_w_vals) if cos_w_vals else None

    # --- null band (random unit vectors in head_dim) -------------------------
    null = random_unit_null_cos(head_dim, args.n_null, args.seed)
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    null_lo_single = float(np.percentile(null, 2.5))
    null_hi_single = float(np.percentile(null, 97.5))
    # Null band for the MEAN over the number of heads the decision metric uses.
    n_for_band = (summary_whitened["n"] if summary_whitened else summary_raw["n"])
    rng = np.random.default_rng(args.seed + 1)
    null_means = np.array([
        np.mean(rng.choice(null, size=max(n_for_band, 1), replace=True))
        for _ in range(20000)
    ])
    null_lo_mean = float(np.percentile(null_means, 2.5))
    null_hi_mean = float(np.percentile(null_means, 97.5))
    print(f"  null (random-unit, D={head_dim}): single-draw mean {null_mean:+.5f} "
          f"sd {null_sd:.5f}  (~0 by concentration; sd≈1/sqrt(D)="
          f"{1.0/np.sqrt(head_dim):.5f})")
    print(f"  null band (mean over n={n_for_band}): "
          f"[{null_lo_mean:+.5f}, {null_hi_mean:+.5f}]", flush=True)

    # --- BCa CIs -------------------------------------------------------------
    bca_raw = bca_ci(np.asarray(cos_raw_vals), args.n_boot, args.seed)
    print(f"\n  cos_raw      mean {summary_raw['mean']:+.4f}  "
          f"BCa95 [{bca_raw[0]:+.4f}, {bca_raw[2]:+.4f}]  (n={summary_raw['n']})")
    bca_w = None
    if summary_whitened:
        bca_w = bca_ci(np.asarray(cos_w_vals), args.n_boot, args.seed)
        print(f"  cos_whitened mean {summary_whitened['mean']:+.4f}  "
              f"BCa95 [{bca_w[0]:+.4f}, {bca_w[2]:+.4f}]  (n={summary_whitened['n']})")

    # --- whitening conditioning ----------------------------------------------
    cond_report = None
    if summary_whitened:
        nzeros = [r.get("n_near_zero_eig", 0) for r in elig if "n_near_zero_eig" in r]
        conds = [r.get("cond_number") for r in elig if r.get("cond_number") is not None]
        cond_report = {
            "n_heads_with_near_zero_eig": int(sum(1 for z in nzeros if z > 0)),
            "total_near_zero_eig_clamped": int(sum(nzeros)),
            "cond_number_median": float(np.median(conds)) if conds else None,
            "cond_number_max": float(np.max(conds)) if conds else None,
            "note": ("high-dim (head_dim 256/512) low-n (~600 pairs) whitening can be "
                     "ill-conditioned; eigvals < 1e-6 clamped (park_whiten convention). "
                     "Many clamped eigvals => whitened cos partly reflects the floor."),
        }
        print(f"\n  whitening conditioning: "
              f"{cond_report['n_heads_with_near_zero_eig']} heads had clamped eigvals; "
              f"median cond {cond_report['cond_number_median']}", flush=True)

    # --- v integrity cross-check (saved v vs recomputed massmean) ------------
    recompute_errs = [r["v_inf_recompute_maxabs_err"] for r in elig
                      if "v_inf_recompute_maxabs_err" in r]
    integrity = None
    if recompute_errs:
        integrity = {
            "v_inf_recompute_maxabs_err_max": float(np.max(recompute_errs)),
            "note": "saved v_inf should equal fit_massmean(mu_correct, mu_wrong); "
                    "max abs err near 0 confirms the extractor's direction matches "
                    "the canonical mass-mean (sanity, not a gate).",
        }
        print(f"  integrity: max |v_inf_saved - massmean(mu)| = "
              f"{integrity['v_inf_recompute_maxabs_err_max']:.2e}", flush=True)

    # --- theta_mc cross-check (§134 bridge) ----------------------------------
    cross_check = None
    tmc_raw = [r["cos_inf_thetamc_raw"] for r in elig
               if np.isfinite(r.get("cos_inf_thetamc_raw", np.nan))]
    if tmc_raw:
        tmc_w = [r["cos_inf_thetamc_whitened"] for r in elig
                 if np.isfinite(r.get("cos_inf_thetamc_whitened", np.nan))]
        cross_check = {
            "n_heads_with_theta_mc": len(tmc_raw),
            "cos_inf_thetamc_raw": _summarize(tmc_raw),
            "cos_inf_thetamc_whitened": _summarize(tmc_w) if tmc_w else None,
            "interpretation": ("§134 bridge: high positive cos(v_inf, theta_mc) => the "
                               "trained mc offset (alpha*theta) ≈ the functional "
                               "inference-direction shift (offset story validated). "
                               "Near 0 => offset is NOT the inference direction."),
        }
        print(f"\n  cross-check cos(v_inf, theta_mc) raw: "
              f"mean {cross_check['cos_inf_thetamc_raw']['mean']:+.4f} "
              f"(n={len(tmc_raw)})", flush=True)
    else:
        print("\n  cross-check: theta_mc absent / no aligned (layer,head) — skipped.",
              flush=True)

    # --- VERDICT (pre-reg §2) ------------------------------------------------
    print("\n" + "=" * 64)
    if has_whitened:
        verdict, gate, rationale = decide_verdict(
            "cos_whitened", bca_w[0], summary_whitened["mean"], bca_w[2],
            null_lo_mean, null_hi_mean)
        decision_metric = "cos_whitened (common metric)"
    elif summary_whitened:
        # Some eligible heads lack covariance -> whitened on a subset only. HELD.
        verdict_w, _, rationale_w = decide_verdict(
            "cos_whitened[subset]", bca_w[0], summary_whitened["mean"], bca_w[2],
            null_lo_mean, null_hi_mean)
        verdict = f"PROVISIONAL_WHITENED_SUBSET_{verdict_w}"
        gate = "HELD_PARTIAL_COVARIANCE"
        rationale = (f"Whitened cos available for only {summary_whitened['n']}/{n_elig} "
                     f"eligible heads (others lack stored covariance). GATE HELD. "
                     f"Provisional read on the covariance-available subset: " + rationale_w)
        decision_metric = "cos_whitened (subset — covariance-available heads only)"
    else:
        verdict_raw, _, rationale_raw = decide_verdict(
            "cos_raw", bca_raw[0], summary_raw["mean"], bca_raw[2],
            null_lo_mean, null_hi_mean)
        verdict = f"PROVISIONAL_RAW_{verdict_raw}"
        gate = "HELD_NEEDS_COVARIANCE"
        rationale = ("WHITENED metric unavailable (no covariance for eligible heads). "
                     "Pre-reg makes whitened the decision metric, so the GATE is HELD. "
                     "Provisional RAW read only: " + rationale_raw)
        decision_metric = "cos_raw (provisional — whitened required for final gate)"

    # Low-N guard: a decisive GREEN/RED gate needs enough eligible heads. With few
    # heads the null band widens (mean over small n) and biases toward ORTHOGONAL,
    # which would spuriously greenlight an expensive process-LoFiT corpus.
    n_decision = summary_whitened["n"] if summary_whitened else 0
    if gate in ("GREEN", "RED") and n_decision < args.min_elig_gate:
        rationale = (f"[HELD_LOW_N] only {n_decision} eligible heads with whitened cos "
                     f"(< --min-elig-gate {args.min_elig_gate}); the '{verdict}' gate is "
                     f"UNDERPOWERED (null band too wide at small n). Lower --acc-threshold "
                     f"to admit more heads or extract more pairs before trusting the gate. "
                     f"Provisional read: " + rationale)
        gate = "HELD_LOW_N"

    print(f"  DECISION METRIC: {decision_metric}")
    print(f"  VERDICT: {verdict}")
    print(f"  process-LoFiT GATE: {gate}")
    print(f"  RATIONALE: {rationale}")
    print("=" * 64, flush=True)

    # --- persist -------------------------------------------------------------
    result = {
        "acts": str(args.acts),
        "schema_version": schema,
        "base": payload.get("base"),
        "chat_template": payload.get("chat_template"),
        "retrieval_datasets_used": payload.get("retrieval_datasets_used"),
        "retrieval_fallback_fired": payload.get("retrieval_fallback_fired"),
        "acc_threshold": args.acc_threshold,
        "extractor_elig_acc": extractor_elig_acc,
        "head_dim": head_dim,
        "n_heads_total": len(records),
        "n_eligible": n_elig,
        "n_eligible_with_whitened": (summary_whitened["n"] if summary_whitened else 0),
        "n_cov_missing": n_cov_missing,
        "has_whitened_all_eligible": has_whitened,
        "decision_metric": decision_metric,
        "verdict": verdict,
        "process_lofit_gate": gate,
        "rationale": rationale,
        "cos_raw": {
            "summary": summary_raw,
            "bca95": {"lo": bca_raw[0], "mean": bca_raw[1], "hi": bca_raw[2]},
        },
        "cos_whitened": ({
            "summary": summary_whitened,
            "bca95": {"lo": bca_w[0], "mean": bca_w[1], "hi": bca_w[2]},
            "metric": "common = (0.5*(cov_inf+cov_ret))^{-1/2}",
        } if summary_whitened else None),
        "null": {
            "head_dim": head_dim,
            "single_draw": {"mean": null_mean, "sd": null_sd,
                            "ci95": [null_lo_single, null_hi_single]},
            "mean_over_decision_n": {"n": n_for_band,
                                     "ci95": [null_lo_mean, null_hi_mean]},
        },
        "whitening_conditioning": cond_report,
        "v_integrity_check": integrity,
        "theta_mc_cross_check": cross_check,
        "records": records,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  wrote {out_path}")

    # Compact stdout summary (parsed by the orchestrator).
    print(json.dumps({
        "path": str(out_path),
        "verdict": verdict,
        "gate": gate,
        "n_eligible": n_elig,
        "n_eligible_with_whitened": (summary_whitened["n"] if summary_whitened else 0),
        "cos_whitened_mean": (summary_whitened["mean"] if summary_whitened else None),
        "cos_raw_mean": summary_raw["mean"],
        "has_whitened_all_eligible": has_whitened,
    }, indent=2))


if __name__ == "__main__":
    main()
