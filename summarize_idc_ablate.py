#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, os, re, sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# ---------- Parse folder name ----------
def parse_folder_params(dirname: str) -> Dict[str, Any]:
    """
    Parse: idc_tr2000_te2000_imb1_a0.1_b1.5_k32
    """
    base = os.path.basename(dirname.rstrip("/"))
    pat = r"idc_tr(?P<tr>\d+)_te(?P<te>\d+)_imb(?P<imb>[0-9.]+)_a(?P<alpha>[0-9.]+)_b(?P<beta>[0-9.]+)_k(?P<kproto>[0-9.]+)"
    m = re.match(pat, base)
    if not m:
        return {}
    out = m.groupdict()
    out["tr"]      = int(float(out["tr"]))
    out["te"]      = int(float(out["te"]))
    out["imb"]     = float(out["imb"])
    out["alpha"]   = float(out["alpha"])
    out["beta"]    = float(out["beta"])
    out["kproto"]  = float(out["kproto"])
    return out

# ---------- Read CSVs ----------
def read_csv_one_row(path: Path, want_method="GM", rename: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row.get("method") == want_method:
                out = dict(row)
                if rename:
                    for src, dst in rename.items():
                        if src in out:
                            out[dst] = out.pop(src)
                # numeric fields
                for k in list(out.keys()):
                    if k in ("acc","nll","brier","ece","smooth","t_fit","t_pred","K","n_train","n_test"):
                        try:
                            out[k] = float(out[k])
                        except Exception:
                            pass
                return out
    return None

def collect_runs(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for d in sorted(root.glob("idc_tr*_te*_imb*_a*_b*_k*")):
        if not d.is_dir():
            continue
        params = parse_folder_params(d.name)
        if not params:
            continue

        cal = read_csv_one_row(d / "idc_calibration.csv", want_method="GM")
        sm  = read_csv_one_row(d / "idc_smooth.csv",      want_method="GM", rename={"smooth":"smooth"})
        summ= read_csv_one_row(d / "idc_summary.csv",     want_method="GM")

        if not cal and not sm and not summ:
            continue

        rec = dict(
            folder=str(d),
            **params,
            acc=None, nll=None, brier=None, ece=None,
            smooth=None, K=None, t_fit=None, t_pred=None, n_train=None, n_test=None
        )
        if cal:
            rec["acc"]   = cal.get("acc", rec["acc"])
            rec["nll"]   = cal.get("nll", rec["nll"])
            rec["brier"] = cal.get("brier", rec["brier"])
            rec["ece"]   = cal.get("ece", rec["ece"])
        if sm:
            rec["smooth"] = sm.get("smooth", rec["smooth"])
        if summ:
            if rec["acc"] is None and "acc" in summ:
                rec["acc"] = summ["acc"]
            rec["K"]       = summ.get("K", rec["kproto"])
            rec["t_fit"]   = summ.get("t_fit", None)
            rec["t_pred"]  = summ.get("t_pred", None)
            rec["n_train"] = summ.get("n_train", None)
            rec["n_test"]  = summ.get("n_test", None)
        if rec["K"] is None:
            rec["K"] = rec["kproto"]
        rows.append(rec)
    return rows

# ---------- Ranking rules ----------
def best_by_calibration(rows: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    cand = [r for r in rows if r.get("ece") is not None]
    if not cand: return None
    return sorted(cand, key=lambda r: (
        r["ece"],                     # min ECE
        r.get("K", float("inf")),     # smaller K
        r.get("nll", float("inf")),   # lower NLL
        r.get("brier", float("inf")), # lower Brier
        - (r.get("acc", -1.0)),       # higher Acc
    ))[0]

def best_by_accuracy(rows: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    cand = [r for r in rows if r.get("acc") is not None]
    if not cand: return None
    return sorted(cand, key=lambda r: (
        - r["acc"],                   # max Acc
        r.get("K", float("inf")),     # smaller K
        r.get("nll", float("inf")),   # lower NLL
        r.get("ece", float("inf")),   # lower ECE
        r.get("brier", float("inf")), # lower Brier
    ))[0]

def best_by_smoothness(rows: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    cand = [r for r in rows if r.get("smooth") is not None]
    if not cand: return None
    return sorted(cand, key=lambda r: (
        r["smooth"],                  # min Smooth
        r.get("K", float("inf")),     # smaller K
        - (r.get("acc", -1.0)),       # higher Acc
        r.get("ece", float("inf")),   # lower ECE
        r.get("nll", float("inf")),   # lower NLL
    ))[0]

# ---------- Writing ----------
def write_csv(path: Path, rows: List[Dict[str,Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in rows:
            wr.writerow([r.get(c, "") for c in cols])

def print_top(rows, key, reverse=False, n=10, label=""):
    print(f"\nTop {n} by {label}:")
    cols_show = ["folder","acc","ece","nll","brier","smooth","K","alpha","beta","kproto","imb","tr","te"]
    rows_sorted = sorted([r for r in rows if r.get(key) is not None],
                         key=lambda r: r[key],
                         reverse=reverse)[:n]
    for i, r in enumerate(rows_sorted, 1):
        summary = "  ".join(f"{c}={r.get(c)}" for c in cols_show if r.get(c) is not None)
        print(f"{i:2d}. {summary}")

# ---------- Grouping (tr==te only) ----------
def group_key_tr_eq_te_imb(r: Dict[str,Any]) -> Optional[str]:
    tr, te = r.get("tr"), r.get("te")
    imb = r.get("imb")
    if tr is None or te is None or imb is None:
        return None
    if tr != te:
        return None
    # normalize imb like "imb1" or "imb8"
    if float(imb).is_integer():
        imb_str = f"imb{int(imb)}"
    else:
        imb_str = f"imb{imb}"
    return f"tr{tr}_te{te}_{imb_str}"

def summarize_group_best(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    For each (tr==te, imb) group, compute best by ECE, Acc, Smoothness.
    Returns a list of rows with extra field 'best_metric' in {'ece','acc','smooth'}.
    """
    groups = {}
    for r in rows:
        k = group_key_tr_eq_te_imb(r)
        if k is None:  # ignore tr!=te for per-group summaries
            continue
        groups.setdefault(k, []).append(r)

    winners = []
    for gkey, grp in groups.items():
        be = best_by_calibration(grp)
        ba = best_by_accuracy(grp)
        bs = best_by_smoothness(grp)
        for rec, tag in ((be,"ece"), (ba,"acc"), (bs,"smooth")):
            if rec is None: 
                continue
            row = dict(rec)  # copy
            row["group"] = gkey
            row["best_metric"] = tag
            winners.append(row)
    return winners

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root dir with idc_tr*_te*_imb*_a*_b*_k* folders")
    ap.add_argument("--save-all", type=str, default="summary_all_gm.csv")
    ap.add_argument("--save-overall", type=str, default="best_overall_gm.csv")
    ap.add_argument("--save-groups", type=str, default="best_per_group_gm.csv")
    args = ap.parse_args()

    root = Path(args.root)
    rows = collect_runs(root)
    if not rows:
        print(f"No runs found under {root}", file=sys.stderr)
        sys.exit(2)

    # Save all rows
    cols_all = ["folder","tr","te","imb","alpha","beta","kproto","K",
                "acc","nll","brier","ece","smooth","t_fit","t_pred","n_train","n_test"]
    write_csv(Path(args.save_all), rows, cols_all)
    print(f"Saved all GM runs → {Path(args.save_all).resolve()}")

    # Overall winners
    best_ece = best_by_calibration(rows)
    best_acc = best_by_accuracy(rows)
    best_smo = best_by_smoothness(rows)
    overall_rows = []
    if best_ece: 
        r = dict(best_ece); r["best_metric"]="ece"; overall_rows.append(r)
    if best_acc:
        r = dict(best_acc); r["best_metric"]="acc"; overall_rows.append(r)
    if best_smo:
        r = dict(best_smo); r["best_metric"]="smooth"; overall_rows.append(r)
    write_csv(Path(args.save_overall), overall_rows, cols_all + ["best_metric"])
    print(f"Saved overall bests → {Path(args.save_overall).resolve()}")

    # Per-group winners (tr==te only)
    group_rows = summarize_group_best(rows)
    write_csv(Path(args.save_groups), group_rows, ["group"] + cols_all + ["best_metric"])
    print(f"Saved per-group bests → {Path(args.save_groups).resolve()}")

    # Console leaderboards (optional)
    print_top(rows, key="ece",     reverse=False, n=10, label="ECE (lower is better)")
    print_top(rows, key="acc",     reverse=True,  n=10, label="Accuracy (higher is better)")
    print_top(rows, key="smooth",  reverse=False, n=10, label="Smoothness (lower is better)")


if __name__ == "__main__":
    main()
