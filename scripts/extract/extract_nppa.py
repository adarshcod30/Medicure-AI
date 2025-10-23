#!/usr/bin/env python3
"""
extract_nppa.py

Robust extractor for NPPA Compendium PDF (ceiling prices).
Primary attempt: camelot (table extraction).
Fallback: pdfplumber + regex row parsing (line-by-line).
Output: CSV with columns:
generic_name, drug, form, strength, unit_qty, unit_name, price, notif, date, source_page, raw_line
"""

import re
import sys
from pathlib import Path
import pandas as pd

PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "Compendium-Prices-2022.pdf"
OUT_CSV  = Path(__file__).resolve().parents[2] / "data" / "processed" / "nppa_ceiling_prices_raw.csv"

# Regex row pattern used in fallback (adapted to NPPA table style)
FORM_WORDS = (
    "Tablet|Capsule|Oral liquid|Oral Liquid|Injection|Suppository|Drops|"
    "Eye Drops|Nasal drops|Nasal Spray|Cream|Ointment|Gel|Lotion|"
    "Powder for Injection|Powder for oral liquid|Solution|Suspension|"
    "Syrup|Pessary|Granules|Enema|Powder|Topical|Inhalation|MDI/DPI|Patch|Sachet|Dry Syrup"
)
STRENGTH = r"[\d\.]+(?:\s?(?:mcg|mg|g|IU|%))(?:\/\s?(?:mL|ml|5 mL|15 mL|actuation|dose))?"
UNIT = r"(Tablet|Capsule|mL|ml|Dose|Vial|Bottle|Ampoule|Pouch|gm|g|Strip|Patch|Sachet)"
DATE = r"(?:\d{2}\.\d{2}\.\d{4}|\d{2}\.\d{2}\.\d{2}|\d{2}\-\d{2}\-\d{4})"

ROW_RE = re.compile(
    rf"^(?P<drug>[A-Za-z0-9\-\(\)\/\+\.,&\s]+?)\s+"
    rf"(?P<form>{FORM_WORDS})\s+"
    rf"(?P<strength>{STRENGTH})\s+"
    rf"(?P<unit_qty>\d+)\s+"
    rf"(?P<unit_name>{UNIT})\s+"
    rf"(?P<price>\d+\.\d+|\d+)\s+"
    rf"(?P<notif>[A-Za-z0-9\/\-\(\)]+)\s+"
    rf"(?P<date>{DATE})$"
)

RELEASE_MARKERS = {"sr", "mr", "er", "xl", "xr"}

def try_camelot_extract(pdf_path):
    try:
        import camelot
    except Exception as e:
        print("[INFO] camelot not available:", e)
        return None

    print("[INFO] Attempting camelot extraction (lattice/stream)...")
    all_frames = []
    try:
        # Try stream flavor first (works for many NPPA tables)
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        print(f"[INFO] Camelot found {len(tables)} tables (stream).")
        for t in tables:
            df = t.df.copy()
            # Heuristic: tables with >4 columns are more likely to be our price tables
            if df.shape[1] >= 6:
                all_frames.append(df)
        # if nothing good, try lattice
        if not all_frames:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
            print(f"[INFO] Camelot found {len(tables)} tables (lattice).")
            for t in tables:
                df = t.df.copy()
                if df.shape[1] >= 6:
                    all_frames.append(df)
    except Exception as e:
        print("[WARN] camelot extraction failed:", e)
        return None

    if not all_frames:
        print("[INFO] camelot did not produce suitable tables.")
        return None

    # Convert frames to a single normalized dataframe (best-effort)
    rows = []
    for df in all_frames:
        # attempt to unify rows: join columns into single line and run fallback parser
        for _, r in df.iterrows():
            line = " ".join([str(x) for x in r.values]).strip()
            rows.append({"source": "camelot", "raw_line": line})
    return pd.DataFrame(rows)

def pdfplumber_fallback(pdf_path):
    import pdfplumber
    print("[INFO] Using pdfplumber fallback extraction...")
    rows = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for line in text.splitlines():
                original = line.strip()
                if not original: 
                    continue
                m = ROW_RE.match(original)
                if m:
                    d = m.groupdict()
                    d["source_page"] = i + 1
                    d["raw_line"] = original
                    rows.append(d)
    if not rows:
        print("[WARN] pdfplumber fallback found 0 matches with regex. Try adjusting ROW_RE or manual inspection.")
        return pd.DataFrame()
    return pd.DataFrame(rows)

def postprocess_df(df):
    # If camelot output DataFrame (one column raw_line), attempt to parse each raw_line with ROW_RE
    if "raw_line" in df.columns and "drug" not in df.columns:
        parsed = []
        for _, r in df.iterrows():
            line = str(r["raw_line"])
            m = ROW_RE.match(line)
            if m:
                d = m.groupdict()
                d["raw_line"] = line
                parsed.append(d)
        if parsed:
            df = pd.DataFrame(parsed)
        else:
            # Nothing parsed; return empty
            return pd.DataFrame()
    # normalize
    if df.empty:
        return df
    df["drug_raw"] = df["drug"]
    # remove section numbers like "2.1.3" at start
    df["drug"] = df["drug"].astype(str).apply(lambda s: re.sub(r"^\d+(\.\d+)*\s*", "", s).strip())
    df["generic_name"] = df["drug"].astype(str).str.lower().str.strip()
    df["form"] = df["form"].astype(str).str.title().str.strip()
    df["unit_name"] = df["unit_name"].astype(str).str.title().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["unit_qty"] = pd.to_numeric(df["unit_qty"], errors="coerce")
    # fix rows where release markers (SR/MR) were extracted alone by merging with previous valid drug
    fixed_drugs = []
    last_drug = None
    for name in df["drug"].astype(str):
        base = name.strip()
        if base.lower() in RELEASE_MARKERS and last_drug:
            base = f"{last_drug} {base}"
        else:
            last_drug = base
        fixed_drugs.append(base)
    df["drug"] = fixed_drugs
    df["generic_name"] = df["drug"].str.lower().str.strip()
    # reorder columns
    cols = ["generic_name","drug","form","strength","unit_qty","unit_name","price","notif","date","source_page","raw_line","drug_raw"]
    available = [c for c in cols if c in df.columns]
    return df[available]

def main():
    pdf = PDF_PATH
    if not pdf.exists():
        print("[ERROR] PDF not found at:", pdf)
        sys.exit(1)
    print("[INFO] PDF found at:", pdf)

    # Try camelot first
    camelot_df = try_camelot_extract(pdf)
    if camelot_df is not None and not camelot_df.empty:
        print("[INFO] Camelot produced a candidate dataset; attempting to parse lines...")
        parsed = postprocess_df(camelot_df)
        if not parsed.empty:
            print("[INFO] Successfully parsed camelot output. Rows:", len(parsed))
            parsed.to_csv(OUT_CSV, index=False)
            print("[DONE] Wrote:", OUT_CSV)
            return
        else:
            print("[WARN] Camelot output could not be parsed into structured rows - falling back to pdfplumber.")
    # Fallback
    pdfplumber_df = pdfplumber_fallback(pdf)
    parsed2 = postprocess_df(pdfplumber_df)
    if not parsed2.empty:
        print("[INFO] Successfully parsed pdfplumber output. Rows:", len(parsed2))
        parsed2.to_csv(OUT_CSV, index=False)
        print("[DONE] Wrote:", OUT_CSV)
        return
    print("[ERROR] Extraction failed. Try opening PDF manually and checking table layout or adjusting ROW_RE regex in the script.")

if __name__ == "__main__":
    main()
