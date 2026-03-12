"""
MediCure AI — OCR Pipeline
Advanced image pre-processing + Tesseract OCR with multi-angle extraction
for pharmaceutical packaging (blister packs, strips, boxes).
"""

import cv2
import numpy as np
import pytesseract
from typing import Tuple
import re


def preprocess_image(image: np.ndarray) -> list[np.ndarray]:
    """
    Apply a robust preprocessing pipeline optimized for reflective
    pharmaceutical packaging.
    
    Returns multiple processed variants for multi-strategy OCR.
    """
    variants = []

    # 1. Grayscale conversion
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. CLAHE — Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Non-Local Means denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=12, templateWindowSize=7, searchWindowSize=21)

    # 4. Standard adaptive threshold variant
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    variants.append(adaptive)

    # 5. Otsu thresholding with inversion (for shiny/reflective surfaces)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 6. Inverted Otsu (catches light text on dark backgrounds)
    _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(otsu_inv)

    # 7. Morphological thin-to-bold transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(otsu, kernel, iterations=1)
    variants.append(dilated)

    return variants


def extract_text_at_angle(image: np.ndarray, angle: int) -> Tuple[str, float]:
    """Extract text from a specific rotation angle."""
    if angle != 0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        
        rotated = cv2.warpAffine(image, matrix, (new_w, new_h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = image

    # Use Tesseract with OEM 3 (LSTM + legacy) and PSM 6 (block of text)
    custom_config = r'--oem 3 --psm 6 -l eng'
    
    try:
        data = pytesseract.image_to_data(rotated, config=custom_config, output_type=pytesseract.Output.DICT)
        
        words = []
        confidences = []
        for i, conf in enumerate(data['conf']):
            c = int(conf)
            if c > 0 and data['text'][i].strip():
                words.append(data['text'][i].strip())
                confidences.append(c)
        
        text = ' '.join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text, avg_confidence
    except Exception as e:
        print(f"OCR error at angle {angle}: {e}")
        return "", 0


def run_ocr_pipeline(image_bytes: bytes) -> Tuple[str, float]:
    """
    Full OCR pipeline:
    1. Decode image
    2. Apply preprocessing variants
    3. Extract text at multiple angles (0°, 90°, 270°)
    4. Return best result by confidence
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return "", 0.0

    # Resize if too large (keeps processing fast)
    max_dim = 2000
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Get preprocessed variants
    variants = preprocess_image(image)
    
    # Multi-angle extraction
    angles = [0, 90, 270]
    
    best_text = ""
    best_confidence = 0.0
    
    for variant in variants:
        for angle in angles:
            text, confidence = extract_text_at_angle(variant, angle)
            
            # Clean the text
            text = clean_ocr_text(text)
            
            if confidence > best_confidence and len(text) > 5:
                best_text = text
                best_confidence = confidence
    
    return best_text, best_confidence


def clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts and normalize text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove isolated single characters (common OCR noise)
    text = re.sub(r'\b[^aAiI]\b', '', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[|}{~`]', '', text)
    
    # Normalize whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
