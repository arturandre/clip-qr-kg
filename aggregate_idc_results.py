#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate IDC experiment results across seeds.

Scans OUTROOT recursively for:
  - *summary.csv
  - *calibration.csv (or *calibrate.csv)
  - *smooth.csv

Groups rows by (base_tag, method), where base_tag is `tag` with a trailing
seed suffix `_s<digit+>` removed. Computes mean and std for numeric columns.

Outputs:
  - aggregated_summary.csv
  - aggregated_calibration.csv
  - aggregated_smooth.csv

Usage:
  python scripts/aggregate_idc_results.py --root runs_idc_ablate --outdir runs_idc_ablate
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

SEED_SUFFIX_RE = re.compile(r"(.*)_s\d+$")  # strips trailing _s0, _s1, ...

def _find_files(root: Path, patterns: List[str]) -> List[Path]:
    hits = []
    for pat in patterns:
        hits.extend(root.rglob(pat))
    # De-dup, keep stable order
    uniq = []
    seen = set()
    for p in sorted(hits):
        if p in seen: continue
        uniq.append(p)
        seen.add(p)
    return uniq

def _ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_base_tag(x: str) -> str:
    if not isinstance(x, str):
        return str(x)
    m = SEED_SUFFIX_RE.match(x)
    return m.group(1) if m else x

def _numeric_cols(df: pd.DataFrame, skip: Tuple[str, ...]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in skip: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def _coerce_numeric(df: pd.DataFrame, extra_to_numeric: List[str] = None):
    if extra_to_numeric:
        for c in extra_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")
    # Also try to coerce obvious numeric strings
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df

def _aggregate(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    kind in {"summary","calibration","smooth"}.
    Aggregates numeric columns by mean/std grouped by (base_tag, method).
    """
    if df.empty:
        return df

    # Derive base_tag
    if "tag" in df.columns:
        df["base_tag"] = df["tag"].astype(str).map(_to_base_tag)
    else:
        # fallback: try from path if present
        df["base_tag"] = "unknown"

    # Standardize dtypes
    df = _coerce_numeric(df)

    # Columns to keep as identifiers
    id_cols = ["base_tag", "method"]
    present_id = [c for c in id_cols if c in df.columns]
    if "method" not in present_id:
        # If method missing, inject a default to avoid grouping errors
        df["method"] = "UNKNOWN"
        present_id = ["base_tag", "method"]

    # Numeric columns to aggregate: skip obvious identifiers
    skip = tuple(set(present_id + ["tag"]))
    num_cols = _numeric_cols(df, skip=skip)
    if not num_cols:
        # Nothing to aggregate; just return unique keys
        return df[present_id].drop_duplicates().reset_index(drop=True)

    # Aggregate
    gb = df.groupby(present_id, dropna=False)
    mean_df = gb[num_cols].mean().add_suffix("_mean")
    std_df  = gb[num_cols].std(ddof=1).add_suffix("_std")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()

    # Optional: bring along representative constant columns if they exist (n_train, n_test, K)
    for c in ["n_train","n_test","K"]:
        if c in df.columns:
            const = gb[c].agg(lambda x: x.iloc[0] if (x.nunique(dropna=False)==1) else np.nan)
            out[c] = const.values

    # Reorder columns: ids first, common metrics next
    metric_order = []
    # Try to place accuracy/NLL/etc first if present
    for prefix in ["acc","nll","t_fit","t_pred","brier","ece"]:
        for suff in ["_mean","_std"]:
            col = prefix + suff
            if col in out.columns:
                metric_order.append(col)
    # Add remaining numeric metrics
    for c in out.columns:
        if c in present_id or c in ["n_train","n_test","K"] or c in metric_order:
            continue
        if c.endswith("_mean") or c.endswith("_std"):
            metric_order.append(c)

    out = out[present_id + metric_order + [c for c in ["n_train","n_test","K"] if c in out.columns]]
    return out

def _load_concat(paths: List[Path], expected_cols: List[str] = None) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src__"] = str(p)
            rows.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame(columns=expected_cols or [])
    df = pd.concat(rows, ignore_index=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory containing run outputs")
    ap.add_argument("--outdir", type=str, default=None, help="Where to write aggregated CSVs (default: --root)")
    ap.add_argument("--summary-patterns", nargs="*", default=["*summary.csv"],
                    help="Glob patterns (recursive) for summary files")
    ap.add_argument("--calibration-patterns", nargs="*", default=["*calibration.csv","*calibrate.csv"],
                    help="Glob patterns (recursive) for calibration files")
    ap.add_argument("--smooth-patterns", nargs="*", default=["*smooth.csv","*dirichlet*.csv"],
                    help="Glob patterns (recursive) for smoothness/energy files")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir or args.root).resolve()
    _ensure_outdir(outdir)

    # ---- SUMMARY ----
    sum_paths = _find_files(root, args.summary_patterns)
    sum_df = _load_concat(sum_paths)
    agg_sum = _aggregate(sum_df, kind="summary") if not sum_df.empty else pd.DataFrame()
    sum_out = outdir / "aggregated_summary.csv"
    if not agg_sum.empty:
        agg_sum.to_csv(sum_out, index=False)
        print(f"[OK] Wrote {sum_out}")
    else:
        print("[INFO] No summary files found")

    # ---- CALIBRATION ----
    cal_paths = _find_files(root, args.calibration_patterns)
    cal_df = _load_concat(cal_paths)
    # If you’ve removed Brier/ECE, script still handles extra cols gracefully
    agg_cal = _aggregate(cal_df, kind="calibration") if not cal_df.empty else pd.DataFrame()
    cal_out = outdir / "aggregated_calibration.csv"
    if not agg_cal.empty:
        agg_cal.to_csv(cal_out, index=False)
        print(f"[OK] Wrote {cal_out}")
    else:
        print("[INFO] No calibration files found")

    # ---- SMOOTH / DIRICHLET ENERGY ----
    sm_paths = _find_files(root, args.smooth_patterns)
    sm_df = _load_concat(sm_paths)
    # Expect columns like: tag, method, energy/flatness/whatever name (script will auto-detect numerics)
    agg_sm = _aggregate(sm_df, kind="smooth") if not sm_df.empty else pd.DataFrame()
    sm_out = outdir / "aggregated_smooth.csv"
    if not agg_sm.empty:
        agg_sm.to_csv(sm_out, index=False)
        print(f"[OK] Wrote {sm_out}")
    else:
        print("[INFO] No smooth/dirichlet files found")

if __name__ == "__main__":
    main()
