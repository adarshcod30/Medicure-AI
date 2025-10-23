# save as scripts/merge/inspect_columns.py and run: python scripts/merge/inspect_columns.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
files = {
    "generic": RAW / "generic.csv",
    "A_Z": RAW / "A_Z_medicines_dataset_of_India.csv",
    "medicine_data": RAW / "medicine_data.csv",
    "nppa": RAW / "nppa_ceiling_prices_clean.csv"
}

for k,f in files.items():
    print(f"\n-- {k} -> {f.name} --")
    try:
        df = pd.read_csv(f, nrows=0)
        print("Columns:", list(df.columns))
    except Exception as e:
        print("Error reading file:", e)
