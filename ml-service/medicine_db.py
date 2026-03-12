"""
MediCure AI — Medicine Database
Loads medicine datasets into memory and provides fast Machine Learning search
using scikit-learn TF-IDF and NearestNeighbors for matching OCR-extracted 
names to known medicines + finding generic alternatives.
"""

import csv
import os
import re
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Data paths (relative to repo root)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

GENERIC_CSV = os.path.join(DATA_DIR, "generic.csv")
AZ_MEDICINES_CSV = os.path.join(DATA_DIR, "A_Z_medicines_dataset_of_India.csv")
MASTER_FINAL_CSV = os.path.join(DATA_DIR, "master_medicines_final.csv")


# ---------------------------------------------------------------------------
# In-memory stores & ML Models (populated at startup)
# ---------------------------------------------------------------------------
_generics: list[dict] = []          
_branded: list[dict] = []           
_master: list[dict] = []            
_loaded = False

# Machine Learning models
_branded_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
_branded_nn = NearestNeighbors(n_neighbors=10, metric='cosine')

_generics_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
_generics_nn = NearestNeighbors(n_neighbors=5, metric='cosine')

_master_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
_master_nn = NearestNeighbors(n_neighbors=5, metric='cosine')


def _normalize(text: str) -> str:
    """Lowercase, strip non-alnum (keep spaces), collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_datasets():
    """Load all CSV datasets into memory and train the ML models."""
    global _generics, _branded, _master, _loaded
    if _loaded:
        return

    # 1. Generic / Jan Aushadhi medicines
    if os.path.exists(GENERIC_CSV):
        with open(GENERIC_CSV, newline="", encoding="utf-8") as f:
            _generics = list(csv.DictReader(f))
        
        # Train Generics ML Model
        generic_names = [_normalize(g.get("Generic Name", "")) for g in _generics]
        if generic_names:
            X_gen = _generics_vectorizer.fit_transform(generic_names)
            _generics_nn.fit(X_gen)
        print(f"[MedicineDB] Loaded & trained ML on {len(_generics)} generic medicines")

    # 2. A-Z Branded medicines
    if os.path.exists(AZ_MEDICINES_CSV):
        with open(AZ_MEDICINES_CSV, newline="", encoding="utf-8") as f:
            _branded = list(csv.DictReader(f))
            
        # Train Branded ML Model
        branded_names = [_normalize(m.get("name", "")) for m in _branded]
        if branded_names:
            X_brand = _branded_vectorizer.fit_transform(branded_names)
            _branded_nn.fit(X_brand)
        print(f"[MedicineDB] Loaded & trained ML on {len(_branded)} branded medicines")

    # 3. Master medicines with NPPA pricing
    if os.path.exists(MASTER_FINAL_CSV):
        with open(MASTER_FINAL_CSV, newline="", encoding="utf-8") as f:
            _master = list(csv.DictReader(f))
            
        # Train Master ML Model
        master_names = [_normalize(m.get("brand_clean", m.get("canonical_inn", ""))) for m in _master]
        if master_names:
            X_master = _master_vectorizer.fit_transform(master_names)
            _master_nn.fit(X_master)
        print(f"[MedicineDB] Loaded & trained ML on {len(_master)} master medicines")

    _loaded = True


# ---------------------------------------------------------------------------
# ML Search helpers
# ---------------------------------------------------------------------------

def search_branded(query: str, top_k: int = 5) -> list[dict]:
    """
    Fast ML search over the A-Z branded medicines using TF-IDF and k-NN.
    """
    if not _branded:
        return []

    norm_q = _normalize(query)
    q_vec = _branded_vectorizer.transform([norm_q])
    distances, indices = _branded_nn.kneighbors(q_vec, n_neighbors=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        # Cosine distance to similarity score
        sim = 1.0 - dist
        if sim > 0.1:  # Relaxed threshold for ML model
            med = _branded[idx]
            results.append({
                "name": med.get("name", ""),
                "price": med.get("price(₹)", med.get("price", "")),
                "manufacturer": med.get("manufacturer_name", ""),
                "type": med.get("type", ""),
                "pack_size": med.get("pack_size_label", ""),
                "composition1": med.get("short_composition1", ""),
                "composition2": med.get("short_composition2", ""),
                "is_discontinued": med.get("Is_discontinued", "FALSE"),
                "match_score": round(sim, 3),
            })
    return results


def search_generics(composition: str, top_k: int = 5) -> list[dict]:
    """
    Find Jan Aushadhi generic alternatives for a given composition string
    using the ML trained TF-IDF model.
    """
    if not _generics:
        return []

    norm_q = _normalize(composition)
    q_vec = _generics_vectorizer.transform([norm_q])
    distances, indices = _generics_nn.kneighbors(q_vec, n_neighbors=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        sim = 1.0 - dist
        if sim > 0.1:
            g = _generics[idx]
            results.append({
                "generic_name": g.get("Generic Name", ""),
                "mrp": g.get("MRP", ""),
                "unit_size": g.get("Unit Size", ""),
                "category": g.get("Group Name", ""),
                "match_score": round(sim, 3),
            })
    return results


def search_master(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the master medicines dataset (includes NPPA ceiling prices)
    using the ML trained TF-IDF model.
    """
    if not _master:
        return []

    norm_q = _normalize(query)
    q_vec = _master_vectorizer.transform([norm_q])
    distances, indices = _master_nn.kneighbors(q_vec, n_neighbors=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        sim = 1.0 - dist
        if sim > 0.1:
            m = _master[idx]
            results.append({
                "brand_name": m.get("brand_clean", m.get("brand_raw", "")),
                "canonical_name": m.get("canonical_inn", ""),
                "composition": m.get("composition_raw", ""),
                "manufacturer": m.get("manufacturer", ""),
                "is_generic": m.get("is_generic", ""),
                "product_price": m.get("product_price", ""),
                "nppa_price": m.get("nppa_price", ""),
                "price_flag": m.get("price_flag", ""),
                "ratio_market_to_nppa": m.get("ratio_market_to_nppa", ""),
                "match_score": round(sim, 3),
            })
    return results


def lookup_medicine(query: str) -> dict:
    """
    Combined lookup: finds best branded match, generic alternatives,
    and NPPA pricing info.  Returns a single unified result dict.
    """
    load_datasets()

    branded_matches = search_branded(query, top_k=3)
    master_matches = search_master(query, top_k=3)

    # Use the best branded match's composition to find generics
    composition_query = query
    if branded_matches:
        comp1 = branded_matches[0].get("composition1", "")
        comp2 = branded_matches[0].get("composition2", "")
        if comp1:
            composition_query = f"{comp1} {comp2}".strip()

    generic_alternatives = search_generics(composition_query, top_k=5)

    # Price flag from master data
    price_info = None
    if master_matches:
        best = master_matches[0]
        price_info = {
            "product_price": best.get("product_price", ""),
            "nppa_ceiling_price": best.get("nppa_price", ""),
            "price_flag": best.get("price_flag", ""),
            "ratio": best.get("ratio_market_to_nppa", ""),
        }

    return {
        "branded_matches": branded_matches,
        "generic_alternatives": generic_alternatives,
        "master_matches": master_matches,
        "price_info": price_info,
    }
