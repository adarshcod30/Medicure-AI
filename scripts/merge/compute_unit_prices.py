# compute_unit_prices.py
"""
Heuristic parser to compute product_price_per_unit by:
 - re-joining master entries with raw A_Z rows (pack_size_label)
 - parsing pack counts like "1 x 10 tablets", "10x10", "strip of 10", "10 tablets"
 - computing product_price_per_unit = product_price / total_unit_count
Saves updated master to data/processed/master_medicines_with_unit_prices.csv
"""

import re
import math
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER_F = ROOT / "data" / "processed" / "master_medicines.csv"
AZ_F     = ROOT / "data" / "raw" / "A_Z_medicines_dataset_of_India.csv"
GENERIC_F= ROOT / "data" / "raw" / "generic.csv"
OUT_F    = ROOT / "data" / "processed" / "master_medicines_with_unit_prices.csv"

def canonical_text(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_pack_count(text):
    """
    Return estimated total units in the pack (int) or None if not found.
    Heuristics:
      - '10 x 10 tablets' -> 100
      - '10x10' -> 100
      - 'strip of 10' -> 10
      - '10 tablets' -> 10
      - '1 vial of 5 ml' -> 1 (unit is vial) -> we keep as 1
      - '2 vials of 2 ml' -> 2
      - '100 ml bottle' -> assume 1 bottle (unit qty unknown) => return None
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower().replace(",", " ").strip()

    # pattern a x b  (e.g., 10 x 10)
    m = re.search(r"(\d+)\s*x\s*(\d+)", t)
    if m:
        a = int(m.group(1)); b = int(m.group(2))
        return a * b

    # pattern 10x10 (no spaces)
    m = re.search(r"\b(\d+)[x×](\d+)\b", t)
    if m:
        return int(m.group(1)) * int(m.group(2))

    # 'strip of 10' or 'strip(s) 10'
    m = re.search(r"(?:strip|strip[s]?|stp)\s*(?:of\s*)?(\d+)", t)
    if m:
        return int(m.group(1))

    # 'pack of 10' or 'bottle of 100'
    m = re.search(r"(?:pack|pack of|packaged|pack size|bottle of|bottle)\s*(?:of\s*)?(\d+)", t)
    if m:
        return int(m.group(1))

    # standalone number before unit word like '10 tablets'
    m = re.search(r"\b(\d+)\s+(tablet|tablets|tab|tabs|capsule|capsules|ml|vial|ampoule|strip|sachet)\b", t)
    if m:
        return int(m.group(1))

    # '1 x 2 ml' -> count 1 (vial)
    m = re.search(r"^(\d+)\s*x\s*\d+\s*ml", t)
    if m:
        return int(m.group(1))

    # If '250 mg/5 ml' type, no pack count
    return None

def safe_float(x):
    try:
        return float(str(x).strip().replace("₹","").replace(",",""))
    except Exception:
        return None

def main():
    master = pd.read_csv(MASTER_F, dtype=str).fillna("")
    az_raw = pd.read_csv(AZ_F, dtype=str).fillna("")
    generic_raw = pd.read_csv(GENERIC_F, dtype=str).fillna("")

    # create a brand->pack_size mapping from az_raw using its 'pack_size_label' column
    if "pack_size_label" in az_raw.columns:
        az_raw["brand_clean"] = az_raw["name"].astype(str).apply(canonical_text)
        pack_map = az_raw.set_index("brand_clean")["pack_size_label"].to_dict()
    else:
        pack_map = {}

    # Also map generic Unit Size from generic.csv (if present)
    if "Unit Size" in generic_raw.columns:
        generic_raw["brand_clean"] = generic_raw["Generic Name"].astype(str).apply(canonical_text)
        generic_unit_map = generic_raw.set_index("brand_clean")["Unit Size"].to_dict()
    else:
        generic_unit_map = {}

    # Add columns to master
    master["pack_size_label"] = ""
    master["estimated_pack_count"] = ""
    master["product_price_num"] = master["product_price"].apply(safe_float)
    master["product_price_per_unit"] = None

    for i, row in master.iterrows():
        src = row.get("source_file","")
        brand_clean = canonical_text(row.get("brand_clean",""))
        pack_label = None
        # first prefer A_Z mapping
        if src == "A_Z" and brand_clean in pack_map:
            pack_label = pack_map[brand_clean]
        # else try generic unit
        if not pack_label and brand_clean in generic_unit_map:
            pack_label = generic_unit_map[brand_clean]
        master.at[i, "pack_size_label"] = pack_label or ""

        est_count = parse_pack_count(pack_label or "")
        master.at[i, "estimated_pack_count"] = est_count if est_count is not None else ""
        # compute price per unit if possible
        price = safe_float(row.get("product_price",""))
        if price is not None and est_count:
            master.at[i, "product_price_per_unit"] = float(price) / float(est_count)
        else:
            master.at[i, "product_price_per_unit"] = None

    # save updated master
    master.to_csv(OUT_F, index=False)
    print("Wrote:", OUT_F)
    # summary
    total_with_unit = master["product_price_per_unit"].notna().sum()
    print("Rows with computed product_price_per_unit:", total_with_unit, " / ", len(master))

if __name__ == "__main__":
    main()
