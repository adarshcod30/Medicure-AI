"""
merge_datasets.py

Merge generic.csv + A_Z_medicines_dataset_of_India.csv + medicine_data.csv + nppa_ceiling_prices_clean.csv
Prioritize generics (keep them first), attach brand products to generics using composition signature,
and enrich with NPPA price. Flag low-confidence matches for manual review.

Place input files in data/raw/ and outputs will be written to data/processed/
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

GENERIC_F = RAW / "generic.csv"
AZ_F = RAW / "A_Z_medicines_dataset_of_India.csv"
MED_F = RAW / "medicine_data.csv"
NPPA_F = RAW / "nppa_ceiling_prices_clean.csv"

OUT_MASTER = OUT / "master_medicines.csv"
OUT_REVIEW = OUT / "to_review_matches.csv"

# --- Helpers -------------------------------------------------------
def canonical_text(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)        # remove parentheses
    s = re.sub(r"\b(tm|®|\u00AE)\b", " ", s)
    s = re.sub(r"[^a-z0-9\+\s\-\/\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_strength_re = re.compile(r"(?P<value>[\d]+(?:\.[\d]+)?)\s*(?P<unit>mcg|μg|mg|g|iu|ml|mL|%)", re.I)
def parse_strength(text):
    if not isinstance(text, str): return (None, None, "")
    m = _strength_re.search(text)
    if m:
        val = float(m.group("value"))
        unit = m.group("unit").lower().replace("μg","mcg")
        return (val, unit, text.strip())
    # numeric fallback
    mm = re.search(r"(\d+(?:\.\d+)?)", str(text))
    return (float(mm.group(1)), "mg", text) if mm else (None, None, str(text))

def parse_composition(text):
    # split on + or comma
    if not isinstance(text, str) or text.strip()=="":
        return []
    parts = re.split(r"\s*\+\s*|\s*,\s*", text)
    out = []
    for p in parts:
        p = p.strip()
        val, unit, _raw = parse_strength(p)
        # remove numeric fragments to get name
        name = re.sub(r"[\d\.\s]*mg|mcg|g|iu|ml|mL|%|/5ml|/ml", "", p, flags=re.I)
        name = re.sub(r"[\(\)\d]", "", name).strip()
        name = canonical_text(name)
        out.append({"name": name, "value": val, "unit": unit})
    out = sorted(out, key=lambda x: x["name"] or "")
    return out

def composition_signature(comp_list):
    # returns a tuple signature (name, value, unit) sorted
    sig = []
    for c in comp_list:
        name = c.get("name") or ""
        try:
            val = float(c.get("value")) if c.get("value") is not None and c.get("value") != "" else 0.0
        except Exception:
            val = 0.0
        unit = (c.get("unit") or "").lower()
        sig.append((name, val, unit))
    return tuple(sig)

# --- Load datasets ------------------------------------------------
print("Loading files...")
try:
    gdf = pd.read_csv(GENERIC_F, dtype=str).fillna("")
    az = pd.read_csv(AZ_F, dtype=str).fillna("")
    md = pd.read_csv(MED_F, dtype=str).fillna("")
    nppa = pd.read_csv(NPPA_F, dtype=str).fillna("")
except Exception as e:
    print("[ERROR] reading input files:", e)
    raise

print("Rows:", len(gdf), len(az), len(md), len(nppa))

# --- Prepare file-specific dataframes (use the actual headers you have) -------------
def prepare_generic_df(df):
    # generic.csv columns: ['Sr No', 'Drug Code', 'Generic Name', 'Unit Size', 'MRP', 'Group Name']
    df = df.copy()
    df["source_file"] = "generic"
    df["product_name"] = df.get("Generic Name", "").astype(str)
    df["salt_composition"] = df["product_name"]  # no explicit composition, use name
    df["product_manufactured"] = df.get("Group Name", "")
    df["product_price"] = df.get("MRP", "")
    df["brand_raw"] = df["product_name"].astype(str)
    df["brand_clean"] = df["brand_raw"].apply(canonical_text)
    df["composition_raw"] = df["salt_composition"].astype(str)
    df["composition_parsed"] = df["composition_raw"].apply(parse_composition)
    df["composition_sig"] = df["composition_parsed"].apply(composition_signature)
    df["manufacturer"] = df.get("product_manufactured","")
    df["match_confidence"] = 100.0
    return df

def prepare_az_df(df):
    # A_Z: ['id','name','price(₹)','Is_discontinued','manufacturer_name','type','pack_size_label','short_composition1','short_composition2']
    df = df.copy()
    df["source_file"] = "A_Z"
    df["product_name"] = df.get("name","").astype(str)
    # prefer short_composition1 else 2
    comp1 = df.get("short_composition1","").astype(str)
    comp2 = df.get("short_composition2","").astype(str)
    df["salt_composition"] = comp1.where(comp1!="", comp2)
    df["product_price"] = df.get("price(₹)","")
    df["product_manufactured"] = df.get("manufacturer_name","")
    df["brand_raw"] = df["product_name"].astype(str)
    df["brand_clean"] = df["brand_raw"].apply(canonical_text)
    df["composition_raw"] = df["salt_composition"].astype(str)
    df["composition_parsed"] = df["composition_raw"].apply(parse_composition)
    df["composition_sig"] = df["composition_parsed"].apply(composition_signature)
    df["manufacturer"] = df["product_manufactured"]
    df["match_confidence"] = 0.0
    return df

def prepare_medicine_data_df(df):
    # medicine_data.csv columns: ['sub_category','product_name','salt_composition','product_price','product_manufactured','medicine_desc','side_effects','drug_interactions']
    df = df.copy()
    df["source_file"] = "medicine_data"
    df["product_name"] = df.get("product_name","").astype(str)
    df["salt_composition"] = df.get("salt_composition","").astype(str)
    df["product_price"] = df.get("product_price","")
    df["product_manufactured"] = df.get("product_manufactured","")
    df["brand_raw"] = df["product_name"].astype(str)
    df["brand_clean"] = df["brand_raw"].apply(canonical_text)
    df["composition_raw"] = df["salt_composition"].astype(str)
    df["composition_parsed"] = df["composition_raw"].apply(parse_composition)
    df["composition_sig"] = df["composition_parsed"].apply(composition_signature)
    df["manufacturer"] = df.get("product_manufactured","")
    df["match_confidence"] = 0.0
    return df

print("[INFO] Preparing generic dataframe...")
gdf_p = prepare_generic_df(gdf)
print("[INFO] Preparing A_Z dataframe...")
az_p  = prepare_az_df(az)
print("[INFO] Preparing medicine_data dataframe...")
md_p  = prepare_medicine_data_df(md)

# NPPA normalization: ensure price_per_unit column exists (compute if possible)
if "price_per_unit" not in nppa.columns:
    try:
        nppa["price"] = pd.to_numeric(nppa.get("price",""), errors="coerce")
        nppa["unit_qty"] = pd.to_numeric(nppa.get("unit_qty",""), errors="coerce")
        nppa["price_per_unit"] = nppa["price"] / nppa["unit_qty"]
    except Exception:
        nppa["price_per_unit"] = nppa.get("price","")

# If NPPA has 'strength' parse numeric
if "strength" in nppa.columns:
    nppa[["strength_value","strength_unit","strength_raw"]] = nppa["strength"].apply(lambda t: pd.Series(parse_strength(t)))
else:
    nppa["strength_value"] = None
    nppa["strength_unit"] = None
    nppa["strength_raw"] = ""

nppa["generic_name_clean"] = nppa["generic_name"].astype(str).apply(canonical_text)
nppa["form_norm"] = nppa.get("form","").astype(str).str.lower().str.strip()

# --- Build initial master: start with generics --------------------------------
print("Building master: seed with generic dataset (priority 1)")
master = []
for _, r in gdf_p.iterrows():
    canonical = canonical_text(r.get("salt_composition") or r.get("product_name") or "")
    master.append({
        "canonical_inn": canonical,
        "brand_raw": r["brand_raw"],
        "brand_clean": r["brand_clean"],
        "composition_raw": r["composition_raw"],
        "composition_parsed": r["composition_parsed"],
        "composition_sig": r["composition_sig"],
        "is_generic": True,
        "manufacturer": r.get("product_manufactured",""),
        "product_price": r.get("product_price",""),
        "source_file": r["source_file"],
        "match_confidence": 100.0,
        "match_method": "native-generic",
        "np_price": None,
        "np_price_per_unit": None,
        "np_price_match_confidence": None,
        "nppa_notif": None,
        "nppa_date": None
    })

master_df = pd.DataFrame(master)
print("Master seeded with generics:", len(master_df))

# --- Helper: signature index --------------------------------
sig_to_index = {}
for i, row in master_df.iterrows():
    sig = row["composition_sig"]
    if sig:
        sig_to_index.setdefault(sig, []).append(i)

def attach_or_add(product_row, source_label):
    """
    Try to attach product_row to existing master by:
     1) exact composition signature match
     2) exact canonical name match (high threshold)
     3) fuzzy canonical name match (lower threshold)
    If attached, return index and method; otherwise append as new row.
    """
    comp_sig = product_row["composition_sig"]
    brand_clean = product_row["brand_clean"]
    # 1) Exact signature
    if comp_sig and comp_sig in sig_to_index:
        idx = sig_to_index[comp_sig][0]
        return idx, "exact-signature", 100.0
    # 2) Exact canonical name match with master canonical_inn
    candidates = master_df["canonical_inn"].astype(str).tolist()
    if len(candidates) == 0:
        # no master rows yet (edge case)
        new_idx = len(master_df)
        master_df.loc[new_idx] = {
            "canonical_inn": brand_clean,
            "brand_raw": product_row["brand_raw"],
            "brand_clean": brand_clean,
            "composition_raw": product_row["composition_raw"],
            "composition_parsed": product_row["composition_parsed"],
            "composition_sig": product_row["composition_sig"],
            "is_generic": False,
            "manufacturer": product_row.get("product_manufactured",""),
            "product_price": product_row.get("product_price",""),
            "source_file": source_label,
            "match_confidence": 0.0,
            "match_method": "new-entry",
            "np_price": None,
            "np_price_per_unit": None,
            "np_price_match_confidence": None,
            "nppa_notif": None,
            "nppa_date": None
        }
        if product_row["composition_sig"]:
            sig_to_index.setdefault(product_row["composition_sig"], []).append(new_idx)
        return new_idx, "new-entry", 0.0

    match = process.extractOne(brand_clean, candidates, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= 95:
        idx = master_df[master_df["canonical_inn"]==match[0]].index[0]
        return idx, "exact-name", float(match[1])
    # 3) fuzzy fallback
    if match and match[1] >= 80:
        idx = master_df[master_df["canonical_inn"]==match[0]].index[0]
        return idx, "fuzzy-name", float(match[1])
    # 4) new entry
    new_idx = len(master_df)
    master_df.loc[new_idx] = {
        "canonical_inn": brand_clean,
        "brand_raw": product_row["brand_raw"],
        "brand_clean": brand_clean,
        "composition_raw": product_row["composition_raw"],
        "composition_parsed": product_row["composition_parsed"],
        "composition_sig": product_row["composition_sig"],
        "is_generic": False,
        "manufacturer": product_row.get("product_manufactured",""),
        "product_price": product_row.get("product_price",""),
        "source_file": source_label,
        "match_confidence": 0.0,
        "match_method": "new-entry",
        "np_price": None,
        "np_price_per_unit": None,
        "np_price_match_confidence": None,
        "nppa_notif": None,
        "nppa_date": None
    }
    if product_row["composition_sig"]:
        sig_to_index.setdefault(product_row["composition_sig"], []).append(new_idx)
    return new_idx, "new-entry", 0.0

# --- Merge A_Z then medicine_data -----------------------------------------
print("Merging A_Z dataset...")
for _, r in az_p.iterrows():
    idx, method, conf = attach_or_add(r, "A_Z")
    if method != "new-entry":
        master_df.at[idx, "match_confidence"] = max(master_df.at[idx, "match_confidence"] or 0.0, conf)
        master_df.at[idx, "match_method"] = master_df.at[idx, "match_method"] or method

print("Merging medicine_data dataset...")
for _, r in md_p.iterrows():
    idx, method, conf = attach_or_add(r, "medicine_data")
    if method != "new-entry":
        master_df.at[idx, "match_confidence"] = max(master_df.at[idx, "match_confidence"] or 0.0, conf)
        master_df.at[idx, "match_method"] = master_df.at[idx, "match_method"] or method

print("Master size after merges:", len(master_df))

# --- Attach NPPA prices (best-effort join) --------------------------------
print("Attaching NPPA prices (best-effort)...")
# build lookup from nppa by (generic_clean, form_norm, strength_value)
nppa_lookup = {}
for _, r in nppa.iterrows():
    key = (canonical_text(r.get("generic_name","")), str(r.get("form_norm","")).lower().strip(), str(r.get("strength_value","")).strip())
    nppa_lookup.setdefault(key, []).append(r)

def find_nppa_price_for_row(row):
    cand_names = []
    for comp in row.get("composition_parsed") or []:
        if comp.get("name"):
            cand_names.append(comp.get("name"))
    cand_names.append(row.get("canonical_inn",""))
    for name in cand_names:
        name_clean = canonical_text(name)
        # attempt to extract strength_val if product is single-ingredient and has value
        strength_val = None
        try:
            if row.get("composition_parsed") and len(row["composition_parsed"])==1:
                strength_val = row["composition_parsed"][0].get("value")
        except Exception:
            strength_val = None
        keys_to_try = []
        if strength_val:
            keys_to_try.append((name_clean, "", str(int(strength_val) if float(strength_val).is_integer() else strength_val)))
            keys_to_try.append((name_clean, "", str(strength_val)))
        keys_to_try.append((name_clean, "",""))
        for k in keys_to_try:
            if k in nppa_lookup:
                recs = nppa_lookup[k]
                try:
                    rec = sorted(recs, key=lambda x: float(x.get("price_per_unit") or x.get("price") or 1e9))[0]
                except Exception:
                    rec = recs[0]
                return rec.get("generic_name"), rec.get("price"), rec.get("unit_qty"), 100.0, rec.get("notif"), rec.get("date")
    return None, None, None, None, None, None

for i, row in master_df.iterrows():
    g, price, unit_qty, conf, notif, date = find_nppa_price_for_row(row)
    if g:
        master_df.at[i, "np_price"] = price
        try:
            master_df.at[i, "np_price_per_unit"] = float(price) / float(unit_qty) if unit_qty else float(price)
        except Exception:
            master_df.at[i, "np_price_per_unit"] = price
        master_df.at[i, "np_price_match_confidence"] = conf
        master_df.at[i, "nppa_notif"] = notif
        master_df.at[i, "nppa_date"] = date

# --- Export master and review file ---------------------------------------
to_review = master_df[(master_df["match_confidence"] < 85) | (master_df["composition_sig"].apply(lambda x: len(x)==0))]
to_review = to_review.sort_values(by="match_confidence")

OUT.parent.mkdir(parents=True, exist_ok=True)
master_df.to_csv(OUT_MASTER, index=False)
to_review.to_csv(OUT_REVIEW, index=False)
print("Wrote master:", OUT_MASTER)
print("Wrote review:", OUT_REVIEW)
print("Rows in master:", len(master_df))
print("Rows to review (low confidence):", len(to_review))
