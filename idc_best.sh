#!/usr/bin/env bash
set -euo pipefail

# ===================== Config =====================
PY=${PYTHON:-python}
SCRIPT=${SCRIPT:-scripts/run_idc.py}      # entry script
OUTROOT=${OUTROOT:-"runs_idc_ablate_best"}
BACKBONE=${BACKBONE:-"resnet18"}
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-256}
IMG_SIZE=${IMG_SIZE:-96}
CACHE_DIR=${CACHE_DIR:-".cache_idc"}

# Fixed setup for this experiment
SAMPLES_PER_CLASS=(2000)                  # 2k train per class, 2k test per class
ALPHA=(0.5)                               # same as synthetic
BETA=(0.1)                                # same as synthetic
KPROTO=(32)                               # GM with 32 prototypes

# Seeds: 5 runs (0..4) for mean/std
SEEDS=(0 1 2 3 4)

# Parallelization (optional). Set NJOBS>1 if GNU parallel is available.
NJOBS=${NJOBS:-1}

# Dry-run: set to 1 to just print commands
DRYRUN=${DRYRUN:-0}

# ===================== Helpers =====================
run_one () {
  local seed="$1" trpc="$2" tepc="$3" a="$4" b="$5" kp="$6"

  local tag="idc_tr${trpc}_te${tepc}_a${a}_b${b}_k${kp}_s${seed}"
  local outdir="${OUTROOT}/${tag}"

  # Skip if a done marker exists
  if [[ -f "${outdir}/_DONE" ]]; then
    echo "[SKIP] ${tag} (found ${outdir}/_DONE)"
    return 0
  fi

  mkdir -p "${outdir}"

  local cmd=(
    "$PY" "$SCRIPT"
    --outdir "$outdir"
    --seed "$seed"
    --cache-dir "$CACHE_DIR"
    --backbone "$BACKBONE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH"
    --train-per-class "$trpc"
    --test-per-class "$tepc"
    --alpha "$a"
    --beta "$b"
    --kproto "$kp"
    --budget-match
  )

  echo "[RUN] ${tag}"
  if [[ "$DRYRUN" -eq 1 ]]; then
    printf '  %q' "${cmd[@]}"; echo
    return 0
  fi

  if "${cmd[@]}"; then
    touch "${outdir}/_DONE"
  else
    echo "[FAIL] ${tag}" >&2
    return 1
  fi
}

export -f run_one
export PY SCRIPT OUTROOT BACKBONE EPOCHS BATCH CACHE_DIR DRYRUN

# ===================== Sweep =====================
# Build the cartesian product of all settings × seeds
build_jobs () {
  for seed in "${SEEDS[@]}"; do
    for spc in "${SAMPLES_PER_CLASS[@]}"; do
      for a in "${ALPHA[@]}"; do
        for b in "${BETA[@]}"; do
          for kp in "${KPROTO[@]}"; do
            echo "$seed $spc $spc $a $b $kp"
          done
        done
      done
    done
  done
}

if [[ "$NJOBS" -gt 1 ]] && command -v parallel >/dev/null 2>&1; then
  # Run with GNU parallel
  build_jobs | parallel -j "$NJOBS" --colsep ' ' run_one {1} {2} {3} {4} {5} {6}
else
  # Sequential fallback
  while read -r seed trpc tepc imb a b kp; do
    run_one "$seed" "$trpc" "$tepc" "$imb" "$a" "$b" "$kp"
  done < <(build_jobs)
fi

echo "All IDC runs done. Outputs under: $OUTROOT"

# ===================== (Optional) Quick aggregation hint =====================
# To aggregate mean/std later, you can do something like:
#   python - <<'PY'
#   import glob, csv, numpy as np, json, os
#   rows=[]
#   for fp in glob.glob(os.path.join(os.environ.get("OUTROOT","runs_idc_ablate"),"*/idc_summary.csv")):
#       with open(fp) as f:
#           r=list(csv.DictReader(f))
#           rows+=r
#   # filter by method, compute mean/std of acc and nll if you also saved calibration CSVs
#   # ...
#   PY
