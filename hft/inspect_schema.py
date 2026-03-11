"""
Drop this in the hft/ directory and run:
    python3 /tmp/inspect_schema.py --data-dir ./data
Prints the full schema of every parquet file found.
"""
import argparse, glob, sys
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="./data")
args = parser.parse_args()

files = sorted(Path(args.data_dir).glob("*.parquet"))
print(f"\nFound {len(files)} parquet files in {args.data_dir}\n")

for f in files:
    df = pd.read_parquet(f)
    print(f"{'='*60}")
    print(f"FILE : {f.name}")
    print(f"ROWS : {len(df):,}")
    print(f"COLS : {list(df.columns)}")
    print(f"TYPES:")
    for col, dtype in df.dtypes.items():
        sample = df[col].iloc[0] if len(df) else "N/A"
        print(f"       {col:<20} {str(dtype):<12} sample={repr(sample)[:80]}")
    print()
