#!/usr/bin/env bash
set -euo pipefail

# ===================== Config =====================
PY=${PYTHON:-python}
SCRIPT=${SCRIPT:-scripts/run_idc.py}   # <-- change to your entry script
OUTROOT=${OUTROOT:-"runs_idc_ablate"}
SEED=${SEED:-0}
BACKBONE=${BACKBONE:-"resnet18"}      # only used if your script needs it
EPOCHS=${EPOCHS:-5}                    # head epochs if training; 0 to skip
BATCH=${BATCH:-256}
CACHE_DIR=${CACHE_DIR:-".cache_idc"}

# Sweeps
SAMPLES_PER_CLASS=(2000 5000 10000 20000)
# TRAIN_PER_CLASS=(2000 5000 10000 20000)
# TEST_PER_CLASS=(2000 5000 10000 20000)
IMBALANCE=(1 8)
ALPHA=(0.4 0.8)
BETA=(0.01 0.3 0.7 0.9)
KPROTO=(32 64 128 256 512)

# Parallelization (optional). Set NJOBS>1 if GNU parallel is available.
NJOBS=${NJOBS:-1}

# Dry-run: set to 1 to just print commands
DRYRUN=${DRYRUN:-0}

# ===================== Helpers =====================
run_one () {
  local trpc="$1" tepc="$2" imb="$3" a="$4" b="$5" kp="$6"

  local tag="idc_tr${trpc}_te${tepc}_imb${imb}_a${a}_b${b}_k${kp}"
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
    --seed "$SEED"
    --cache-dir "$CACHE_DIR"
    --backbone "$BACKBONE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH"
    --train-per-class "$trpc"
    --test-per-class "$tepc"
    --imbalance "$imb"
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
export PY SCRIPT OUTROOT SEED BACKBONE EPOCHS BATCH CACHE_DIR DRYRUN

# ===================== Sweep =====================
# Build the cartesian product of all settings
build_jobs () {
  for spc in "${SAMPLES_PER_CLASS[@]}"; do
    for imb in "${IMBALANCE[@]}"; do
      for a in "${ALPHA[@]}"; do
        for b in "${BETA[@]}"; do
          for kp in "${KPROTO[@]}"; do
            echo "$spc $spc $imb $a $b $kp"
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
  while read -r spc spc imb a b kp; do
    run_one "$spc" "$spc" "$imb" "$a" "$b" "$kp"
  done < <(build_jobs)
fi

echo "All ablations done. Outputs under: $OUTROOT"
