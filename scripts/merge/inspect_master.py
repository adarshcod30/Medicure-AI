# inspect_master.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "master_medicines.csv"
REVIEW = ROOT / "data" / "processed" / "to_review_matches.csv"

def main():
    df = pd.read_csv(MASTER, dtype=str).fillna("")
    review = pd.read_csv(REVIEW, dtype=str).fillna("")

    print("Master rows:", len(df))
    print("Review rows (low confidence):", len(review))
    print()

    # How many have NPPA price attached?
    np_attached = df['np_price'].replace("", pd.NA).notna().sum()
    print(f"Rows with NPPA price attached: {np_attached} ({np_attached/len(df):.1%})")

    # Top 20 canonical_inn by count (i.e., how many brands per generic)
    top = df.groupby("canonical_inn").size().sort_values(ascending=False).head(20)
    print("\nTop 20 canonical INNs (brands / entries):")
    print(top.to_string())

    # For generics vs non-generics
    gen_count = (df["is_generic"].astype(str).str.lower() == "true").sum()
    print(f"\nIs_generic=True rows: {gen_count}")

    # Show 10 sample low-confidence rows
    print("\nSample to_review (first 10):")
    print(review[["canonical_inn","brand_raw","composition_raw","match_confidence","match_method"]].head(10).to_string(index=False))

    # Show sample rows where no composition_sig
    no_comp = df[df["composition_sig"].apply(lambda x: x == "()" or x=="[]") | df["composition_sig"].isna()]
    print(f"\nRows with empty composition signature: {len(no_comp)}")
    if len(no_comp)>0:
        print(no_comp[["brand_raw","source_file","composition_raw"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
