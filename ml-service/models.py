"""
MediCure AI — Pydantic Models
Request / response schemas for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# /analyze response
# ---------------------------------------------------------------------------

class AlternativeMedicine(BaseModel):
    name: str = ""
    price: str = ""
    source: str = ""   # "jan_aushadhi", "generic", "branded"
    drug_code: str = ""


class MedicineAnalysis(BaseModel):
    """Structured output from the /analyze endpoint."""
    brand_name: Optional[str] = ""
    generic_name: Optional[str] = ""
    composition: Optional[str] = ""
    form: Optional[str] = ""               # tablet, syrup, capsule, injection …
    manufacturer: Optional[str] = ""
    indications: list[str] = Field(default_factory=list) # detailed list of what it cures
    uses: list[str] = Field(default_factory=list)        # plain english uses
    side_effects: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list) # exactly what it should NOT be taken with
    interactions: list[str] = Field(default_factory=list)
    schedule_type: Optional[str] = ""      # e.g. "Schedule H", "Schedule G", "OTC"
    dosage: Optional[str] = ""
    storage: Optional[str] = ""
    warnings: list[str] = Field(default_factory=list)
    price: Optional[str] = ""
    nppa_ceiling_price: Optional[str] = ""
    price_flag: Optional[str] = ""         # "ok", "overpriced", "missing"
    cheaper_alternatives: list[AlternativeMedicine] = Field(default_factory=list)
    simplified_explanation: Optional[str] = ""
    confidence: float = 0.0
    ocr_text: Optional[str] = ""
    method: Optional[str] = "ocr"          # "ocr" | "vision_fallback"


class AnalyzeResponse(BaseModel):
    success: bool = True
    data: Optional[MedicineAnalysis] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    scan_id: str = ""
    message: str
    target_language: str = "en-US"            # "en-US", "hi-IN", "ta-IN", etc.
    medicine_context: Optional[dict] = None   # sent from backend
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    success: bool = True
    reply: str = ""
    error: Optional[str] = None
