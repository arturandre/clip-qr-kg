#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Graph Memory (GM) — Toy Experiments (Binary)
# ================================================================
# Runs all clean + long-tail configurations for:
#   - datasets: moons, circles
#   - seeds:    0,1,2 (customizable)
#   - modes:    clean, longtail
#
# Each run executes the full baseline + GM suite in the Python script:
#   Linear, kNN, LabelSpreading, LP-harmonic, GM, GM-flat,
#   GM-instance, GM-centroid, GM→LP(instance), kNN-budget (optional)
#
# Example usage:
#   ./run_toy_all.sh
# ================================================================

# -------- Config --------
PY=python
SCRIPT=scripts/run_gm.py                # Path to your Python experiment script
OUTROOT="runs_toy_best_idc2"            # Root output directory
SEEDS=(0 1 2 3 4)                       # Random seeds
IMBALANCE=8                             # Long-tail ratio (set to 1 for balanced)
KNN_BUDGET=1                            # 1=enable budget-matched kNN; 0=disable
PARALLEL=0                              # 1=run clean/longtail in parallel per dataset

# -------- Checks --------
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python not found in PATH." >&2
  exit 1
fi

if [ ! -f "$SCRIPT" ]; then
  echo "Cannot find Python script: $SCRIPT" >&2
  exit 1
fi

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

# -------- Helper --------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_block () {
  local dataset="$1"
  local tag="$2"
  local outdir="$3"
  local seed="$4"
  local imbalance_flag=()
  local budget_flag=()

  if [ "$tag" = "longtail" ]; then
    imbalance_flag=(--imbalance "$IMBALANCE")
  fi
  if [ "$KNN_BUDGET" -eq 1 ]; then
    budget_flag=(--budget-match)
  fi

  echo "[$(timestamp)]  ==> ${dataset} | ${tag} | seed=${seed}"
  $PY "$SCRIPT" \
    --dataset "$dataset" \
    --outdir  "$outdir" \
    --seed    "$seed" \
    "${imbalance_flag[@]}" \
    "${budget_flag[@]}"
}

# -------- Runs --------
for dataset in moons circles; do
  for seed in "${SEEDS[@]}"; do
    CLEAN_OUT="${OUTROOT}/${dataset}/clean/s${seed}"
    LONG_OUT="${OUTROOT}/${dataset}/longtail/s${seed}"

    if [ "$PARALLEL" -eq 1 ]; then
      run_block "$dataset" "clean"    "$CLEAN_OUT" "$seed" &
      run_block "$dataset" "longtail" "$LONG_OUT"  "$seed" &
      wait
    else
      run_block "$dataset" "clean"    "$CLEAN_OUT" "$seed"
      run_block "$dataset" "longtail" "$LONG_OUT"  "$seed"
    fi
  done
done

# -------- Aggregation --------
echo
echo "[$(timestamp)]  Aggregating all CSV summaries..."
cl=("clean" "longtail")
for dataset in moons circles; do
  for dtype in "${cl[@]}"; do
    SUMPATH="${OUTROOT}/${dataset}/${dtype}"
    AGG_CSV="${OUTROOT}/all_summaries_${dataset}_${dtype}.csv"
    {
      echo "tag,method,acc,t_fit,t_pred,n_train,n_test,K"
      find "$SUMPATH" -type f -name "*_summary.csv" -exec tail -n +2 {} \;
    } > "$AGG_CSV"

    AGG_CAL_CSV="${OUTROOT}/all_calibrations_${dataset}_${dtype}.csv"
    {
      echo "tag,method,acc,nll,brier,ece"
      find "$SUMPATH" -type f -name "*_calibration.csv" -exec tail -n +2 {} \;
    } > "$AGG_CAL_CSV"

    AGG_SMO_CSV="${OUTROOT}/all_smooths_${dataset}_${dtype}.csv"
    {
      echo "tag,method,smooth"
      find "$SUMPATH" -type f -name "*_smooth.csv" -exec tail -n +2 {} \;
    } > "$AGG_SMO_CSV"
  done
done

echo "[$(timestamp)]  ✅ All toy runs completed."
echo "[$(timestamp)]  Aggregated results → ${OUTROOT}/all_sumaries*.csv"
echo "[$(timestamp)]  Aggregated calibrations → ${OUTROOT}/all_calibrations*.csv"
echo "[$(timestamp)]  Aggregated smooths → ${OUTROOT}/all_smooths*.csv"
echo "[$(timestamp)]  Outputs per run   → ${OUTROOT}"
