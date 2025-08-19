# Medicure-AI# Medicure-AI 🤖💊  
**AI/ML Intelligence for Meddy Buddy — An AI-Powered Medical Assistant**  

---

## 📌 Overview  
This repository contains the **AI/ML components** of the **Meddy Buddy (Medicure)** project.  
It powers the intelligent backend of the system, including:  

- 🔎 **Medicine Identification** from pill images & packaging (vision + OCR fusion).  
- 📄 **Medical Report OCR & Simplification** (NER + NLP).  
- 🩺 **Symptom-to-Condition Suggestions** with conservative triage guidance.  
- 💰 **Generic & Affordable Alternatives** (via Jan Aushadhi dataset).  
- 🌐 **Multilingual Support** (Indian languages + text-to-speech).  

The **web/UI layer** is maintained separately in [`medicure-web`](https://github.com/your-org/medicure-web).  

---

## ⚙️ Features in This Repo  
- **Pill Recognition Pipeline** (CNN + OCR + fusion scoring).  
- **Report Processing**: OCR + medical NER + plain-language simplification.  
- **Symptom Engine**: Hybrid Naive Bayes + Knowledge Graph scoring.  
- **Language Support**: IndicTrans2 + TTS for Hindi and regional languages.  

---

## 📊 Datasets  
- **Pill Images**: NIH Pillbox, RxImage + locally collected Indian pill images.  
- **Drug Knowledge Base**: Jan Aushadhi (PMBJP), CDSCO drug lists.  
- **Symptom–Condition Mapping**: SymCat, WHO, ICD-10 + curated clinical rules.  
- **Reports**: Synthetic + de-identified medical reports.  

---

## 🛠️ Tech Stack  
- **ML/DL:** PyTorch, Hugging Face Transformers, Scikit-learn.  
- **OCR:** Tesseract, EasyOCR.  
- **Serving:** TorchServe, ONNX Runtime.  
- **Database:** MongoDB (drug KB, conditions).  
- **API Layer:** FastAPI (inference endpoints).  

---

## 🧪 Training & Evaluation  
- **Pill Classifier**: EfficientNet/MobileNet with data augmentation.  
- **Metrics**:  
  - Top-k accuracy (vision model).  
  - CER (Character Error Rate) for OCR.  
  - Precision/Recall/F1 for symptom triage.  
  - Inference latency target: ≤ 2 seconds.  

---

## 🚀 Getting Started  

### 1. Clone repo  
```bash
git clone https://github.com/your-org/medicure-ai.git
cd medicure-ai
