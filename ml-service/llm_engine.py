"""
MediCure AI — LLM Inference Engine
Uses Google Gemini 1.5 Pro for:
  1. Semantic cleaning and structuring of OCR text
  2. Vision Fallback (send raw image when OCR confidence is low)
  3. Contextual chatbot responses
"""

import json
import os
import re
import base64
import requests
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

_model = None
_vision_model = None


def _get_model():
    global _model
    if _model is None:
        _model = genai.GenerativeModel(MODEL_NAME)
    return _model


def _get_vision_model():
    global _vision_model
    if _vision_model is None:
        _vision_model = genai.GenerativeModel(MODEL_NAME)
    return _vision_model


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYZE_SYSTEM_PROMPT = """You are MediCure AI, a pharmaceutical analysis assistant.

Given the raw OCR text extracted from a medicine package (and optionally a database lookup result), produce a STRUCTURED JSON analysis.

## Rules
1. **Identify** the medicine: brand name, generic/salt name, composition with strengths.
2. **Simplify** all medical jargon into plain English a non-medical person can understand.
3. **Include** uses, indications (what it cures exactly), side effects, precautions, contraindications (what it should NOT be taken with / harmful combinations), drug interactions, dosage guidance, and storage instructions.
4. **Flag warnings** (pregnancy, driving, alcohol, kidney/liver impairment).
5. Identify the **schedule classification** (e.g., Schedule H, Schedule G, Schedule H1, OTC).
6. If database matches are provided, use them for accurate pricing and generic alternatives.
7. If the text is fragmented/noisy, do your best to reconstruct the medicine name.
8. Do NOT invent information. If unsure, say "Not available on packaging".

## Output — STRICT JSON (no markdown fences):
{
  "brand_name": "...",
  "generic_name": "...",
  "composition": "...",
  "form": "tablet|capsule|syrup|injection|cream|...",
  "manufacturer": "...",
  "indications": ["exact disease or condition 1", "..."],
  "uses": ["plain English use 1", "..."],
  "side_effects": ["plain English effect 1", "..."],
  "precautions": ["...", "..."],
  "contraindications": ["exact harmful combination or condition 1", "..."],
  "interactions": ["...", "..."],
  "schedule_type": "Schedule H etc...",
  "dosage": "...",
  "storage": "...",
  "warnings": ["...", "..."],
  "simplified_explanation": "A 2-3 sentence plain English summary of what this medicine is and what it does.",
  "confidence": 0.0-1.0
}"""

VISION_SYSTEM_PROMPT = """You are MediCure AI. The user has uploaded a photo of medicine packaging but OCR could not reliably read the text.

Look at the image and identify:
1. The medicine name (brand and generic if visible)
2. Composition / active ingredients
3. Manufacturer
4. Any other text you can read (dosage, warnings, Schedule H/G etc.)

Return the SAME structured JSON as the text analysis endpoint.

## Output — STRICT JSON (no markdown fences):
{
  "brand_name": "...",
  "generic_name": "...",
  "composition": "...",
  "form": "...",
  "manufacturer": "...",
  "indications": ["..."],
  "uses": ["..."],
  "side_effects": ["..."],
  "precautions": ["..."],
  "contraindications": ["..."],
  "interactions": ["..."],
  "schedule_type": "...",
  "dosage": "...",
  "storage": "...",
  "warnings": ["..."],
  "simplified_explanation": "...",
  "confidence": 0.0-1.0
}"""

CHAT_SYSTEM_PROMPT = """You are MediCure AI, a friendly and knowledgeable pharmaceutical assistant.

You are chatting with a user about a specific medicine they scanned. Here is the medicine context:
{context}

## Rules
1. Answer ONLY questions related to this medicine or general pharmaceutical topics.
2. Use SIMPLE, plain English — avoid medical jargon.
3. If the user asks about dosage changes, always recommend consulting a doctor.
4. Be concise but thorough.
5. If you genuinely don't know, say so — never invent medical advice.
6. You may suggest cheaper generic alternatives from Jan Aushadhi if relevant.
7. Always include a brief medical disclaimer when giving health-related advice."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    # Try to find JSON in code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1)

    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return {}


async def analyze_text(
    ocr_text: str,
    db_matches: Optional[dict] = None,
    target_language: str = "en-US",
) -> dict:
    """
    Send OCR text (+ optional database matches) to Gemini for
    structured medicine analysis.
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not configured"}

    model = _get_model()

    user_prompt = f"OCR Text from medicine packaging:\n\n{ocr_text}"
    if db_matches:
        user_prompt += f"\n\nDatabase lookup results:\n{json.dumps(db_matches, indent=2, default=str)}"

    if target_language != "en-US":
        user_prompt += f"\n\nCRITICAL TRANSLATION INSTRUCTION: Translate ALL text values (brand_name, generic_name, composition, forms, arrays like indications, side_effects, interactions, warnings, precautions, explanations, etc.) into the language code '{target_language}'. The ONLY things that remain in English are the JSON keys."

    try:
        response = model.generate_content(
            [ANALYZE_SYSTEM_PROMPT, user_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        return _extract_json(response.text)
    except Exception as e:
        print(f"[LLM] Text analysis error: {e}")
        return {"error": str(e)}


async def analyze_image(image_bytes: bytes, target_language: str = "en-US") -> dict:
    """
    Vision Fallback: send the raw image to Gemini multimodal
    when OCR confidence is too low.
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not configured"}

    model = _get_vision_model()

    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes,
    }

    prompt_parts = [VISION_SYSTEM_PROMPT]
    if target_language != "en-US":
        lang_instruction = f"\n\nCRITICAL TRANSLATION INSTRUCTION: Translate ALL text values (brand_name, generic_name, composition, forms, arrays like indications, side_effects, interactions, warnings, precautions, explanations, etc.) into the language code '{target_language}'. The ONLY things that remain in English are the JSON keys."
        prompt_parts.append(lang_instruction)
        
    prompt_parts.append(image_part)

    try:
        response = model.generate_content(
            prompt_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        return _extract_json(response.text)
    except Exception as e:
        print(f"[LLM] Vision analysis error: {e}")
        return {"error": str(e)}


async def chat_response(
    message: str,
    medicine_context: Optional[dict] = None,
    history: Optional[list[dict]] = None,
) -> str:
    """
    Contextual chatbot response about a scanned medicine.
    """
    if not GEMINI_API_KEY:
        return "I'm sorry, the AI service is not configured. Please set your GEMINI_API_KEY."

    model = _get_model()

    context_str = json.dumps(medicine_context, indent=2, default=str) if medicine_context else "No medicine context available."
    system = CHAT_SYSTEM_PROMPT.format(context=context_str)

    # Build conversation
    contents = [system]
    if history:
        for msg in history[-10:]:  # keep last 10 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            contents.append(f"{role}: {content}")

    contents.append(f"user: {message}")

    try:
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        return response.text
    except Exception as e:
        print(f"[LLM] Chat error: {e}")
        return f"I encountered an error processing your question. Please try again. ({e})"


# ---------------------------------------------------------------------------
# Multilingual / Translation Engine (HuggingFace + Gemini Fallback)
# ---------------------------------------------------------------------------
HF_API_TOKEN = os.getenv("SARVAM_API_KEY", "")  # User's HuggingFace token

# Language code → HuggingFace Helsinki-NLP model mapping
HF_TRANSLATION_MODELS = {
    "hi-IN": "Helsinki-NLP/opus-mt-en-hi",
    "ta-IN": "Helsinki-NLP/opus-mt-en-ta",  # English → Tamil (via mul)
    "te-IN": "Helsinki-NLP/opus-mt-en-te",  # English → Telugu (via mul)
    "bn-IN": "Helsinki-NLP/opus-mt-en-bn",  # English → Bengali (via mul)
    "mr-IN": "Helsinki-NLP/opus-mt-en-mr",  # English → Marathi (via mul)
}
# Fallback multi-target model for unsupported languages
HF_FALLBACK_MODEL = "Helsinki-NLP/opus-mt-en-mul"


async def translate_text(text: str, target_language: str = "hi-IN") -> str:
    """
    Translates text into the target Indian language.
    Primary: HuggingFace Inference API (Helsinki-NLP models).
    Fallback: Gemini.
    """
    if target_language == "en-US":
        return text

    # 1. Try HuggingFace Inference API
    if HF_API_TOKEN and HF_API_TOKEN.startswith("hf_"):
        model_id = HF_TRANSLATION_MODELS.get(target_language, HF_FALLBACK_MODEL)
        url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        # Split long text into chunks (HF models have token limits)
        chunks = [text[i:i+450] for i in range(0, len(text), 450)]
        translated_parts = []

        try:
            for chunk in chunks:
                payload = {"inputs": chunk}
                resp = requests.post(url, json=payload, headers=headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        translated_parts.append(data[0].get("translation_text", chunk))
                    else:
                        translated_parts.append(chunk)
                else:
                    print(f"[Translate] HuggingFace returned {resp.status_code}: {resp.text}")
                    translated_parts.append(chunk)
            result = " ".join(translated_parts)
            if result.strip():
                return result
        except Exception as e:
            print(f"[Translate] HuggingFace failed: {e}. Falling back to Gemini.")

    # 2. Fallback to Gemini Translation
    if not GEMINI_API_KEY:
        return text

    model = _get_model()
    prompt = f"Translate the following medical text into {target_language}. Ensure the medical meaning is preserved but simply explained. Return ONLY the translated text without any conversational filler:\n\n{text}"

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Translate] Gemini Fallback failed: {e}")
        return text
