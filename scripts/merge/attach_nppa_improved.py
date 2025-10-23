"""
attach_nppa_improved.py

Progressive NPPA price attachment:
- tries exact composition signature (if signature exists in master and we computed NPPA signatures)
- exact canonical INN match (normalized)
- exact generic_name match (NPPA)
- fuzzy match (token_sort_ratio) on names (thresholds)
- if single-ingredient product, also tries to match by strength (numeric tolerance)

Input:
  data/processed/master_medicines_with_unit_prices.csv  (if exists) else data/processed/master_medicines.csv
  data/raw/nppa_ceiling_prices_clean.csv

Output:
  data/processed/master_medicines_with_nppa.csv
  data/processed/nppa_unmatched_top.csv
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

ROOT = Path(__file__).resolve().parents[2]
MASTER_IN1 = ROOT / "data" / "processed" / "master_medicines_with_unit_prices.csv"
MASTER_IN2 = ROOT / "data" / "processed" / "master_medicines.csv"
NPPA_F = ROOT / "data" / "raw" / "nppa_ceiling_prices_clean.csv"
OUT_MASTER = ROOT / "data" / "processed" / "master_medicines_with_nppa.csv"
OUT_UNMATCHED = ROOT / "data" / "processed" / "nppa_unmatched_top.csv"

def canonical_text(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\+\s\-\/\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_strength_re = re.compile(r"(?P<value>[\d]+(?:\.[\d]+)?)\s*(?P<unit>mcg|μg|mg|g|iu|ml|mL|%)", re.I)
def parse_strength_val(text):
    if not isinstance(text, str): return None, None
    m = _strength_re.search(text)
    if not m: return None, None
    val = float(m.group("value"))
    unit = m.group("unit").lower().replace("μg","mcg")
    return val, unit

def load_master():
    if MASTER_IN1.exists():
        return pd.read_csv(MASTER_IN1, dtype=str).fillna("")
    else:
        return pd.read_csv(MASTER_IN2, dtype=str).fillna("")

def build_nppa_lookup(nppa_df):
    """
    Build several lookups:
     - by (generic_clean, form_norm, strength_value) => rows
     - by generic_clean => rows (list)
     Also keep variants for fuzzy matching lists
    """
    nppa_df = nppa_df.copy()
    nppa_df["generic_clean"] = nppa_df["generic_name"].astype(str).apply(canonical_text)
    nppa_df["form_norm"] = nppa_df.get("form","").astype(str).str.lower().str.strip()
    # parse strength numeric if present
    if "strength" in nppa_df.columns:
        nppa_df[["strength_value","strength_unit"]] = nppa_df["strength"].apply(lambda t: pd.Series(parse_strength_val(str(t))))
    else:
        nppa_df["strength_value"] = None
        nppa_df["strength_unit"] = None
    # ensure price_per_unit present
    if "price_per_unit" not in nppa_df.columns:
        try:
            nppa_df["price"] = pd.to_numeric(nppa_df.get("price",""), errors="coerce")
            nppa_df["unit_qty"] = pd.to_numeric(nppa_df.get("unit_qty",""), errors="coerce")
            nppa_df["price_per_unit"] = nppa_df["price"] / nppa_df["unit_qty"]
        except Exception:
            nppa_df["price_per_unit"] = pd.to_numeric(nppa_df.get("price",""), errors="coerce")

    by_key = {}
    by_generic = {}
    names = []
    for i, r in nppa_df.iterrows():
        key = (r["generic_clean"], str(r.get("form_norm","")).strip(), str(r.get("strength_value") or ""))
        by_key.setdefault(key, []).append(r.to_dict())
        by_generic.setdefault(r["generic_clean"], []).append(r.to_dict())
        names.append(r["generic_clean"])
    return by_key, by_generic, list(set(names)), nppa_df

def try_match_row(row, nppa_by_key, nppa_by_gen, nppa_names):
    """
    Attempt matches in this order
    Return tuple: (matched_record_dict or None, method, score)
    """
    # Try composition signature -> if master had composition_sig and NPPA has entries for component name(s)
    comps = row.get("composition_parsed") or []
    # gather possible generic tokens from composition
    possible_names = [c.get("name") for c in comps if c.get("name")]
    if row.get("canonical_inn"):
        possible_names.append(row["canonical_inn"])
    # normalize
    possible_names = [canonical_text(x) for x in possible_names if x]

    # 1) exact by (generic,form,strength)
    for name in possible_names:
        # try to extract strength if single ingredient
        strength_val = None
        if len(comps)==1:
            strength_val = comps[0].get("value")
        # try keys
        keys = []
        if strength_val:
            keys.append((name, "", str(int(strength_val) if float(strength_val).is_integer() else strength_val)))
            keys.append((name, "", str(strength_val)))
        keys.append((name, "",""))
        for k in keys:
            if k in nppa_by_key:
                recs = nppa_by_key[k]
                # choose lowest price_per_unit if available
                rec = sorted(recs, key=lambda r: float(r.get("price_per_unit") or r.get("price") or 1e9))[0]
                return rec, "exact-key", 100.0

    # 2) exact generic name match
    for name in possible_names:
        if name in nppa_by_gen:
            recs = nppa_by_gen[name]
            rec = sorted(recs, key=lambda r: float(r.get("price_per_unit") or r.get("price") or 1e9))[0]
            return rec, "exact-generic", 98.0

    # 3) direct exact match against NPPA canonical names (edge case)
    for name in possible_names:
        if name in nppa_names:
            # fetch by generic
            recs = nppa_by_gen.get(name, [])
            if recs:
                rec = sorted(recs, key=lambda r: float(r.get("price_per_unit") or r.get("price") or 1e9))[0]
                return rec, "exact-nameset", 96.0

    # 4) fuzzy match on name tokens (token_sort_ratio)
    # Build candidate list from nppa_names and match with RapidFuzz
    for name in possible_names:
        choice = process.extractOne(name, nppa_names, scorer=fuzz.token_sort_ratio)
        if choice and choice[1] >= 85:
            matched_gen = choice[0]
            recs = nppa_by_gen.get(matched_gen, [])
            if recs:
                rec = sorted(recs, key=lambda r: float(r.get("price_per_unit") or r.get("price") or 1e9))[0]
                return rec, "fuzzy-name", float(choice[1])

    # no match
    return None, None, 0.0

def main():
    master = load_master()
    print("Master rows:", len(master))
    nppa_df = pd.read_csv(NPPA_F, dtype=str).fillna("")
    nppa_by_key, nppa_by_gen, nppa_names, nppa_df = build_nppa_lookup(nppa_df)
    print("NPPA entries:", len(nppa_df), "unique names:", len(nppa_names))

    # prepare output columns
    master["np_price_attached"] = False
    master["nppa_match_method"] = ""
    master["nppa_match_confidence"] = 0.0
    master["nppa_price"] = ""
    master["nppa_price_per_unit"] = ""

    unmatched_names = {}

    for i, r in master.iterrows():
        try:
            rec, method, score = try_match_row(r, nppa_by_key, nppa_by_gen, nppa_names)
            if rec:
                master.at[i, "np_price_attached"] = True
                master.at[i, "nppa_match_method"] = method
                master.at[i, "nppa_match_confidence"] = score
                master.at[i, "nppa_price"] = rec.get("price") or rec.get("price_per_unit") or ""
                master.at[i, "nppa_price_per_unit"] = rec.get("price_per_unit") or rec.get("price") or ""
            else:
                # track unmatched canonical_inn for summary
                keyname = canonical_text(r.get("canonical_inn","") or r.get("brand_clean",""))
                unmatched_names[keyname] = unmatched_names.get(keyname, 0) + 1
        except Exception as e:
            # avoid stopping on single-row errors
            print("Row error:", i, e)

    total_attached = master["np_price_attached"].sum()
    print("Total NPPA prices attached:", total_attached, "/", len(master))

    # write outputs
    master.to_csv(OUT_MASTER, index=False)

    # write top unmatched canonical names for manual review (top 200)
    unmatched_sorted = sorted(unmatched_names.items(), key=lambda x: x[1], reverse=True)[:200]
    df_un = pd.DataFrame(unmatched_sorted, columns=["canonical_inn","count"])
    df_un.to_csv(OUT_UNMATCHED, index=False)
    print("Wrote master with NPPA:", OUT_MASTER)
    print("Wrote unmatched summary:", OUT_UNMATCHED)
    return

if __name__ == "__main__":
    main()
