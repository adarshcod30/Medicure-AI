"""
MediCure AI — ML Microservice (FastAPI)
Endpoints:
  POST /analyze  — image upload → OCR → DB lookup → LLM → structured JSON
  POST /chat     — medicine context + user message → conversational reply
  GET  /health   — health check
"""

import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel as PydanticBaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from ocr_pipeline import run_ocr_pipeline
from medicine_db import load_datasets, lookup_medicine
from llm_engine import analyze_text, analyze_image, chat_response, translate_text
from models import (
    MedicineAnalysis,
    AlternativeMedicine,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
)

load_dotenv()

# ---------------------------------------------------------------------------
# App lifecycle — load datasets once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load medicine datasets into memory on startup."""
    print("[MediCure ML] Loading medicine datasets…")
    load_datasets()
    print("[MediCure ML] Datasets ready. Server is live.")
    yield
    print("[MediCure ML] Shutting down.")


app = FastAPI(
    title="MediCure AI — ML Service",
    version="1.0.0",
    description="OCR + LLM pipeline for pharmaceutical packaging analysis",
    lifespan=lifespan,
)

# CORS — allow the Node backend and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
        os.getenv("BACKEND_URL", "http://localhost:3001"),
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Confidence threshold for vision fallback
# ---------------------------------------------------------------------------
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "40"))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "medicure-ml",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


# ---------------------------------------------------------------------------
# POST /analyze — full medicine analysis pipeline
# ---------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...), target_language: str = Form("en-US")):
    """
    Full pipeline:
      1. Read uploaded image
      2. Run OCR pipeline (multi-angle, multi-variant)
      3. If confidence < threshold → vision fallback (send image to Gemini)
      4. Look up extracted text in medicine databases
      5. Send OCR text + DB matches to Gemini for structured analysis
      6. Merge DB pricing / generic alternatives into final result
      7. Return structured AnalyzeResponse
    """
    try:
        # 1. Read image bytes
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        # 2. OCR pipeline
        ocr_text, ocr_confidence = run_ocr_pipeline(image_bytes)
        print(f"[Analyze] OCR confidence: {ocr_confidence:.1f}%, text length: {len(ocr_text)}")

        # 3. Decide: OCR path vs Vision Fallback
        method = "ocr"
        llm_result = {}

        if ocr_confidence < OCR_CONFIDENCE_THRESHOLD or len(ocr_text) < 10:
            # Vision Fallback — let Gemini read the image directly
            print("[Analyze] Low OCR confidence → using Vision Fallback")
            method = "vision_fallback"
            llm_result = await analyze_image(image_bytes, target_language=target_language)
        else:
            # 4. Database lookup from OCR text
            db_matches = lookup_medicine(ocr_text)

            # 5. LLM structured analysis
            llm_result = await analyze_text(ocr_text, db_matches, target_language=target_language)

        if "error" in llm_result:
            return AnalyzeResponse(
                success=False,
                error=llm_result["error"],
            )

        # 6. Build the structured response
        analysis = MedicineAnalysis(
            brand_name=llm_result.get("brand_name", ""),
            generic_name=llm_result.get("generic_name", ""),
            composition=llm_result.get("composition", ""),
            form=llm_result.get("form", ""),
            manufacturer=llm_result.get("manufacturer", ""),
            indications=llm_result.get("indications", []),
            uses=llm_result.get("uses", []),
            side_effects=llm_result.get("side_effects", []),
            precautions=llm_result.get("precautions", []),
            contraindications=llm_result.get("contraindications", []),
            interactions=llm_result.get("interactions", []),
            schedule_type=llm_result.get("schedule_type", ""),
            dosage=llm_result.get("dosage", ""),
            storage=llm_result.get("storage", ""),
            warnings=llm_result.get("warnings", []),
            simplified_explanation=llm_result.get("simplified_explanation", ""),
            confidence=llm_result.get("confidence", 0.0),
            ocr_text=ocr_text,
            method=method,
        )

        # 7. Enrich with database pricing + generic alternatives
        if method == "ocr":
            search_query = analysis.brand_name or analysis.generic_name or ocr_text
            db_data = lookup_medicine(search_query)

            if db_data.get("price_info"):
                pi = db_data["price_info"]
                analysis.price = analysis.price or pi.get("product_price", "")
                analysis.nppa_ceiling_price = pi.get("nppa_ceiling_price", "")
                analysis.price_flag = pi.get("price_flag", "")

            # Add generic alternatives from Jan Aushadhi dataset
            for alt in db_data.get("generic_alternatives", []):
                analysis.cheaper_alternatives.append(
                    AlternativeMedicine(
                        name=alt.get("generic_name", ""),
                        price=f"₹{alt.get('mrp', '')} for {alt.get('unit_size', '')}",
                        source="jan_aushadhi",
                        drug_code=alt.get("drug_code", ""),
                    )
                )

        return AnalyzeResponse(success=True, data=analysis)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return AnalyzeResponse(success=False, error=str(e))


# ---------------------------------------------------------------------------
# POST /chat — contextual chatbot
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Contextual chatbot: takes medicine context + user message,
    returns a conversational LLM response.
    Supports multilingual input/output via Sarvam AI.
    """
    try:
        # 1. Translate user message to English for processing
        message_to_process = request.message
        if request.target_language != "en-US":
            message_to_process = await translate_text(request.message, target_language="en-US")

        # 2. Get LLM Response in English
        reply_en = await chat_response(
            message=message_to_process,
            medicine_context=request.medicine_context,
            history=request.history,
        )

        # 3. Translate response back to user's language
        final_reply = reply_en
        if request.target_language != "en-US":
             final_reply = await translate_text(reply_en, target_language=request.target_language)

        return ChatResponse(success=True, reply=final_reply)

    except Exception as e:
        traceback.print_exc()
        return ChatResponse(success=False, error=str(e))


# ---------------------------------------------------------------------------
# POST /search — text-based medicine name search
# ---------------------------------------------------------------------------
class SearchRequest(PydanticBaseModel):
    name: str
    target_language: str = "en-US"


@app.post("/search", response_model=AnalyzeResponse)
async def search_medicine(request: SearchRequest):
    """
    Text-based medicine search:
      1. Look up medicine name in local datasets
      2. Send name + DB matches to Gemini for structured analysis
      3. Return AnalyzeResponse
    """
    try:
        medicine_name = request.name.strip()
        if not medicine_name:
            return AnalyzeResponse(success=False, error="Medicine name is required")

        # 1. Database lookup
        db_matches = lookup_medicine(medicine_name)

        # 2. LLM structured analysis using medicine name as "OCR text"
        llm_result = await analyze_text(
            f"Medicine name: {medicine_name}",
            db_matches,
            target_language=request.target_language,
        )

        if "error" in llm_result:
            return AnalyzeResponse(success=False, error=llm_result["error"])

        # 3. Build structured response
        analysis = MedicineAnalysis(
            brand_name=llm_result.get("brand_name", medicine_name),
            generic_name=llm_result.get("generic_name", ""),
            composition=llm_result.get("composition", ""),
            form=llm_result.get("form", ""),
            manufacturer=llm_result.get("manufacturer", ""),
            indications=llm_result.get("indications", []),
            uses=llm_result.get("uses", []),
            side_effects=llm_result.get("side_effects", []),
            precautions=llm_result.get("precautions", []),
            contraindications=llm_result.get("contraindications", []),
            interactions=llm_result.get("interactions", []),
            schedule_type=llm_result.get("schedule_type", ""),
            dosage=llm_result.get("dosage", ""),
            storage=llm_result.get("storage", ""),
            warnings=llm_result.get("warnings", []),
            simplified_explanation=llm_result.get("simplified_explanation", ""),
            confidence=llm_result.get("confidence", 0.0),
            ocr_text=medicine_name,
            method="text_search",
        )

        # 4. Enrich with database pricing + generic alternatives
        if db_matches.get("price_info"):
            pi = db_matches["price_info"]
            analysis.price = analysis.price or pi.get("product_price", "")
            analysis.nppa_ceiling_price = pi.get("nppa_ceiling_price", "")
            analysis.price_flag = pi.get("price_flag", "")

        for alt in db_matches.get("generic_alternatives", []):
            analysis.cheaper_alternatives.append(
                AlternativeMedicine(
                    name=alt.get("generic_name", ""),
                    price=f"₹{alt.get('mrp', '')} for {alt.get('unit_size', '')}",
                    source="jan_aushadhi",
                    drug_code=alt.get("drug_code", ""),
                )
            )

        return AnalyzeResponse(success=True, data=analysis)

    except Exception as e:
        traceback.print_exc()
        return AnalyzeResponse(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("ML_SERVICE_PORT", "8000")),
        reload=True,
    )
