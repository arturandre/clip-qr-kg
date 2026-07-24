# FILE: scripts/plot_memcurve_bars.py
"""Minimal bar-plot aggregator for CSV results (model,acc)."""
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

# import os, sys
# # --- make project root importable (../) so 'graphmemory' is found ---
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csvs', nargs='+', required=False, help='CSV files to aggregate')
    ap.add_argument('--out', default='runs/summary.png')
    args = ap.parse_args()
    if not args.csvs:
        print('No CSVs provided; skipping.')
        return
    rows = []
    for path in args.csvs:
        df = pd.read_csv(path)
        df['source'] = os.path.basename(path)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    pivot = df.pivot_table(index='source', columns='model', values='acc')
    ax = pivot.plot(kind='bar', figsize=(8,4))
    plt.ylabel('Accuracy'); plt.title('Baselines vs GraphMemory')
    plt.tight_layout(); plt.savefig(args.out, dpi=180)
    print('Saved', args.out)

if __name__ == '__main__':
    main()
