# emotion_detector.py  (v3 — GoEmotions BERT + IAM style + fusion)
# ============================================================
# Prerequisites (run once before starting Flask):
#   python STEP1_train_text_model.py    → emotion_text_model/
#   python STEP2_train_style_classifier.py → style_classifier.pkl
# ============================================================

import os
import json
import base64
import pickle
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import pytesseract

from feature_extractor import extract_features, features_to_vector, human_readable

# ── Import StyleEnsemble BEFORE any pickle.load so pickle can resolve it ──
from STEP2_train_style_classifier import StyleEnsemble  # noqa: F401

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_DIR  = os.path.join(BASE_DIR, "emotion_text_model")
STYLE_CLF_PATH  = os.path.join(BASE_DIR, "style_classifier.pkl")
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract OCR\tesseract.exe"

EMOTIONS = ["happy", "sad", "angry", "stressed"]

# Fusion weights — text is the only reliable signal from real photos
# Style features are shown in UI but don't affect emotion prediction
TEXT_WEIGHT  = 1.00
STYLE_WEIGHT = 0.00

# ─────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─────────────────────────────────────────────
# Load text model (GoEmotions fine-tuned DistilBERT)
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

def load_text_model():
    # Handle case where zip extracted to a nested folder
    nested = os.path.join(TEXT_MODEL_DIR, "emotion_text_model")
    model_dir = nested if os.path.isdir(nested) else TEXT_MODEL_DIR

    if not os.path.isdir(model_dir) or not os.path.exists(
            os.path.join(model_dir, "config.json")):
        print(f"[WARN] Text model not found at {model_dir}")
        print("       Run STEP1_train_text_model.py first.")
        return None, None, None
    print(f"[INFO] Loading text emotion model from {model_dir}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()
    with open(os.path.join(model_dir, "label_map.json")) as f:
        label_map = json.load(f)
    return tokenizer, model, label_map

text_tokenizer, text_model, label_map = load_text_model()

# Fallback: use original DistilBERT sentiment if STEP1 not run yet
if text_model is None:
    print("[INFO] Falling back to DistilBERT sentiment analysis...")
    from transformers import pipeline as hf_pipeline
    _sentiment_pipe = hf_pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
else:
    _sentiment_pipe = None

# ─────────────────────────────────────────────
# Load style classifier (STEP2)
# ─────────────────────────────────────────────
def load_style_classifier():
    if not os.path.exists(STYLE_CLF_PATH):
        print(f"[WARN] Style classifier not found at {STYLE_CLF_PATH}")
        print("       Run STEP2_train_style_classifier.py first.")
        return None
    print(f"[INFO] Loading style classifier from {STYLE_CLF_PATH}...")
    with open(STYLE_CLF_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload

style_payload = load_style_classifier()
style_model   = style_payload["model"] if style_payload else None

# ─────────────────────────────────────────────
# OCR — EasyOCR (handles cursive/handwriting)
# ─────────────────────────────────────────────
_easyocr_reader = None

def get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[INFO] Loading EasyOCR model (first time only)...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _easyocr_reader


def ocr_extract(img: Image.Image) -> str:
    import cv2
    import numpy as np

    try:
        arr  = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Upscale for better recognition
        h, w = gray.shape
        if w < 1600:
            scale = 1600 / w
            gray  = cv2.resize(gray, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_CUBIC)

        # CLAHE for lighting correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # EasyOCR reads directly from numpy array
        reader  = get_ocr_reader()
        results = reader.readtext(gray, detail=1, paragraph=False)

        # Filter low-confidence results and join
        lines = []
        for (bbox, text, conf) in results:
            if conf > 0.2 and len(text.strip()) > 1:
                # Keep only if majority alphabetic
                alpha = sum(c.isalpha() or c.isspace() for c in text)
                if len(text) > 0 and alpha / len(text) > 0.5:
                    lines.append(text.strip())

        text = " ".join(lines)
        print(f"[DEBUG] OCR extracted {len(text)} chars: '{text[:200]}'")
        return text

    except Exception as e:
        print(f"[WARN] EasyOCR failed: {e}, falling back to Tesseract")
        try:
            gray2 = img.convert("L")
            return pytesseract.image_to_string(gray2).strip()
        except Exception:
            return ""




def _dewarp(gray):
    """
    Detect the page/writing area and correct perspective distortion.
    Falls back to just deskewing if no clear rectangle is found.
    """
    import cv2, numpy as np

    try:
        h, w = gray.shape

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 100)
        edges   = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)

        # Find contours — look for the largest quadrilateral (the page)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        page_quad = None
        for cnt in cnts:
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > (h * w * 0.2):
                page_quad = approx
                break

        if page_quad is not None:
            # Order points: top-left, top-right, bottom-right, bottom-left
            pts  = page_quad.reshape(4, 2).astype(np.float32)
            rect = _order_points(pts)
            tl, tr, br, bl = rect

            # Destination width/height
            dw = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
            dh = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))

            if dw > 100 and dh > 100:
                dst = np.array([[0,0],[dw-1,0],[dw-1,dh-1],[0,dh-1]],
                               dtype=np.float32)
                M   = cv2.getPerspectiveTransform(rect, dst)
                return cv2.warpPerspective(gray, M, (dw, dh))

        # Fallback: just deskew rotation
        return _deskew_simple(gray)

    except Exception:
        return gray


def _order_points(pts):
    import numpy as np
    rect  = np.zeros((4, 2), dtype=np.float32)
    s     = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect


def _deskew_simple(img):
    import cv2, numpy as np
    try:
        _, thresh = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.3 or abs(angle) > 20:
            return img
        h, w   = img.shape
        M      = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return img


# ─────────────────────────────────────────────
# Text emotion (GoEmotions BERT)
# ─────────────────────────────────────────────
def text_emotion_probs(text: str) -> dict:
    """
    Returns probability distribution over 4 emotions from text content.
    Uses fine-tuned GoEmotions model if available, else sentiment fallback.
    """
    if not text or len(text.strip()) < 3:
        # No text: return uniform distribution
        return {e: 0.25 for e in EMOTIONS}

    if text_model is not None:
        # GoEmotions model
        inputs = text_tokenizer(
            text[:512], return_tensors="pt",
            truncation=True, padding=True, max_length=128
        ).to(DEVICE)
        with torch.no_grad():
            logits = text_model(**inputs).logits
        probs_all = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Map from model's label space to our 4 emotions
        # Model may have 5 classes (including neutral)
        id2label  = label_map["id2label"]
        probs_4   = {e: 0.0 for e in EMOTIONS}
        neutral_p = 0.0
        for idx, p in enumerate(probs_all):
            lbl = id2label[str(idx)]
            if lbl in probs_4:
                probs_4[lbl] += float(p)
            else:
                neutral_p += float(p)

        # Redistribute neutral probability proportionally
        total_4 = sum(probs_4.values())
        if total_4 > 0:
            extra_each = neutral_p / 4
            probs_4 = {e: probs_4[e] + extra_each for e in EMOTIONS}
            # Renormalize
            s = sum(probs_4.values())
            probs_4 = {e: probs_4[e] / s for e in EMOTIONS}
        else:
            probs_4 = {e: 0.25 for e in EMOTIONS}

        return probs_4

    else:
        # Fallback: sentiment → simple mapping
        probs_4 = {e: 0.1 for e in EMOTIONS}
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for s in sentences[:10]:
            try:
                result = _sentiment_pipe(s[:512])[0]
                label  = result["label"].lower()
                score  = result["score"]
                lower  = s.lower()
                if label == "positive":
                    probs_4["happy"] += score
                else:
                    if any(w in lower for w in ["angry","hate","rage","fury"]):
                        probs_4["angry"] += score
                    elif any(w in lower for w in ["stress","tired","overwhelm","anxious"]):
                        probs_4["stressed"] += score
                    else:
                        probs_4["sad"] += score
            except Exception:
                continue
        total = sum(probs_4.values())
        return {e: probs_4[e] / total for e in EMOTIONS}


# ─────────────────────────────────────────────
# Style emotion
# ─────────────────────────────────────────────
def style_emotion_probs(img: Image.Image) -> tuple:
    """
    Returns (probs_dict, feature_dict, human_readable_dict).
    """
    try:
        features = extract_features(img)
        vec      = features_to_vector(features)

        if style_model is not None:
            _, _, probs = style_model.predict_single(vec)
        else:
            # Fallback: rule-based from features
            probs = _rule_based_probs(features)

        return probs, features, human_readable(features)

    except Exception as e:
        print(f"[WARN] Style extraction failed: {e}")
        return {e: 0.25 for e in EMOTIONS}, {}, {}


def _rule_based_probs(features: dict) -> dict:
    """
    Simple rule-based fallback when style model is unavailable.
    Based purely on the most discriminative features.
    """
    scores = {e: 0.0 for e in EMOTIONS}
    slant   = features.get("baseline_slant", 0)
    irreg   = features.get("baseline_irregularity", 0)
    sw      = features.get("stroke_width_mean", 0)
    press   = features.get("pressure_mean", 128)
    p_var   = features.get("pressure_std", 0)

    if slant > 2:      scores["happy"]   += 2.0
    if slant < -2:     scores["sad"]     += 2.0
    if press > 155:    scores["angry"]   += 2.0
    if sw > 5.5:       scores["angry"]   += 1.5
    if irreg > 5:      scores["stressed"]+= 2.5
    if p_var > 35:     scores["stressed"]+= 1.5

    # Ensure positive
    scores = {e: max(scores[e], 0.05) for e in EMOTIONS}
    total  = sum(scores.values())
    return {e: scores[e] / total for e in EMOTIONS}


# ─────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────
def fuse(text_probs: dict, style_probs: dict, text: str) -> tuple:
    """
    Fusion: text is the primary emotion signal.
    Style is shown in UI but does not affect the emotion decision
    (style classifier is unreliable on real phone-camera photos).
    If no text extracted, falls back to style with heavy smoothing.
    """
    has_text = len(text.strip()) >= 10

    if has_text:
        # Pure text prediction
        fused = {e: text_probs.get(e, 0.25) for e in EMOTIONS}
    else:
        # No text: use style but smooth heavily toward uniform
        # to avoid false confident predictions
        uniform = 0.25
        smoothing = 0.55  # 55% toward uniform
        fused = {e: smoothing * uniform + (1 - smoothing) * style_probs.get(e, 0.25)
                 for e in EMOTIONS}

    # Normalize
    s = sum(fused.values())
    fused = {e: fused[e] / s for e in EMOTIONS}

    emotion    = max(fused, key=fused.get)
    confidence = round(fused[emotion] * 100, 1)
    probs_pct  = {e: round(fused[e] * 100, 1) for e in EMOTIONS}

    return emotion, confidence, probs_pct


# ─────────────────────────────────────────────
# Mental health risk
# ─────────────────────────────────────────────
def assess_risk(fused_probs: dict, segments: list) -> str:
    # Based on text segments + dominant negative emotion probability
    neg_prob = fused_probs.get("sad", 0) + \
               fused_probs.get("angry", 0) + \
               fused_probs.get("stressed", 0)
    neg_segs = sum(1 for s in segments if s["emotion"] in ["sad","angry","stressed"])

    if neg_prob > 0.70 or neg_segs >= 3:
        return "High"
    if neg_prob > 0.45 or neg_segs >= 2:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────
# Segment-level text analysis
# ─────────────────────────────────────────────
def analyse_segments(text: str) -> list:
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 3]
    results   = []
    for s in sentences[:15]:
        seg_probs = text_emotion_probs(s)
        emotion   = max(seg_probs, key=seg_probs.get)
        results.append({"text": s, "emotion": emotion,
                         "confidence": round(seg_probs[emotion]*100, 1)})
    return results


# ─────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────
def analyse(img: Image.Image) -> dict:
    # 1. OCR
    text = ocr_extract(img)

    # 2. Text emotion
    t_probs = text_emotion_probs(text)

    # 3. Style emotion
    s_probs, raw_features, feat_summary = style_emotion_probs(img)

    # 4. Fusion
    emotion, confidence, fused_probs = fuse(t_probs, s_probs, text)

    # 5. Segment analysis
    segments = analyse_segments(text)

    # 6. Risk
    risk = assess_risk(fused_probs, segments)

    return {
        "emotion":          emotion,
        "confidence":       confidence,
        "probabilities":    fused_probs,
        "text_probs":       {e: round(t_probs[e]*100,1) for e in EMOTIONS},
        "style_probs":      {e: round(s_probs[e]*100,1) for e in EMOTIONS},
        "text":             text,
        "segments":         segments,
        "risk":             risk,
        "feature_summary":  feat_summary,
        "raw_features":     {k: round(float(v),4) for k,v in raw_features.items()},
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    file     = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    img    = Image.open(filepath)
    result = analyse(img)
    result["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result)


@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    data       = request.json["image"]
    img_bytes  = base64.b64decode(data.split(",")[1])
    img        = Image.open(BytesIO(img_bytes))
    result     = analyse(img)
    result["image_url"] = ""
    return jsonify(result)


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)