# QAI Project v3 — Setup Guide
# ============================================================
# Handwriting Emotion Detection
# GoEmotions BERT (text) + IAM-calibrated style features (handwriting)
# ============================================================

## Files in this build
    STEP1_train_text_model.py       ← Fine-tune DistilBERT on GoEmotions
    STEP2_train_style_classifier.py ← Train handwriting style classifier
    feature_extractor.py            ← Image feature extraction (auto-used)
    emotion_detector.py             ← Flask app (main entry point)
    templates/home.html             ← UI (keep existing)

## Step 1 — Install dependencies
    pip install torch torchvision transformers datasets
    pip install scikit-learn opencv-python pillow flask
    pip install pytesseract werkzeug

## Step 2 — Train the text model (run on Colab GPU or local GPU)
    python STEP1_train_text_model.py

    This downloads GoEmotions (~58k sentences, 4 emotions),
    fine-tunes DistilBERT for 4 epochs (~20 mins on Colab T4),
    and saves the model to: emotion_text_model/

    If running on Colab:
      - Upload STEP1_train_text_model.py to Colab
      - Runtime → Change runtime type → T4 GPU
      - Run the script
      - Download the entire emotion_text_model/ folder
      - Place it in D:\QAI Project\emotion_text_model\

## Step 3 — Train the style classifier
    python STEP2_train_style_classifier.py

    Without IAM data: runs automatically with synthetic data (~30 sec)
    With IAM data (better):
      - Register free at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
      - Download formsA-D.tgz, formsE-H.tgz, formsI-Z.tgz
      - Extract all .png files to: D:\QAI Project\data\iam_forms\
      - Then run the script

    Saves: style_classifier.pkl

## Step 4 — Run the app
    python emotion_detector.py

    Open: http://127.0.0.1:5000

## How it works (for your report)

    INPUT IMAGE
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
    STYLE ANALYSIS           CONTENT ANALYSIS
    (handwriting features)   (OCR + NLP)
                             
    OpenCV extracts:         Tesseract OCR extracts text
    - Baseline slant         DistilBERT (fine-tuned on
    - Stroke width           GoEmotions, 58k samples)
    - Letter size            classifies emotion from
    - Pressure proxy         written content
    - Spacing metrics
    - Stroke smoothness
    - Margin consistency
         │                        │
         ▼                        ▼
    StyleEnsemble            GoEmotions BERT
    (RF + GB + MLP)          (5-class → 4-class)
    40% weight               60% weight
         │                        │
         └──────────┬─────────────┘
                    ▼
              Weighted Fusion
                    │
                    ▼
           Emotion + Confidence
           + Risk Assessment
           + Feature Breakdown

## Citations (for your report)
    1. Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions.
       ACL 2020. https://arxiv.org/abs/2005.00547

    2. Marti & Bunke (2002). The IAM-database: an English sentence database
       for offline handwriting recognition. IJDAR 5(1):39-46.

    3. Sanh et al. (2019). DistilBERT, a distilled version of BERT.
       https://arxiv.org/abs/1910.01108

    4. Champa & Rani (2010). Automated Human Behavior Prediction through
       Handwriting Analysis. ICIIP 2010.

    5. Nezos (1993). Graphology: The Art of Handwriting Analysis.
       Scriptor Books.
