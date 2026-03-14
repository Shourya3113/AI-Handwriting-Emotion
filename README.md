# AI-Handwriting-Emotion
AI system that detects emotional state from handwriting images using fine-tuned BERT (GoEmotions) + OpenCV handwriting feature analysis.

AI Enhanced Handwriting Emotion Detection

Detects emotional state (happy, sad, angry, stressed) from handwriting images 
by analysing both written content and handwriting style simultaneously.

## How It Works
- **Text analysis:** EasyOCR extracts text → DistilBERT (fine-tuned on 
  GoEmotions, 58k sentences) classifies emotion from content
- **Style analysis:** OpenCV extracts 13 handwriting features (slant, pressure, 
  stroke width, spacing, etc.) → Ensemble classifier predicts style emotion
- **Fusion:** Both signals combined into final prediction with confidence score 
  and risk level

## Setup
1. pip install -r requirements.txt
2. python STEP1_train_text_model.py  (requires GPU)
3. python STEP2_train_style_classifier.py
4. python emotion_detector.py
5. Open http://127.0.0.1:5000

## Tech Stack
Python, Flask, PyTorch, HuggingFace Transformers, OpenCV, EasyOCR, Scikit-learn

## Datasets
- GoEmotions (Demszky et al., ACL 2020)
- IAM Handwriting Database (Marti & Bunke, 2002)
