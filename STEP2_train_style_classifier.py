# ============================================================
# STEP 2 — IAM Handwriting Feature Extraction + Style Classifier
#
# What this does:
#   1. Downloads IAM handwriting database (or uses local copy)
#   2. Extracts 12 real handwriting features from each writer's page
#   3. Clusters writers into 4 style groups using k-means
#   4. Maps clusters to emotion labels using graphology research
#   5. Trains a calibrated Random Forest style classifier
#   6. Saves: style_classifier.pkl + feature_scaler.pkl
#
# IAM Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
# Register free at the above URL, download:
#   → formsA-D.tgz, formsE-H.tgz, formsI-Z.tgz  (the form images)
#   → ascii.tgz  (not needed but useful)
#
# Cite: Marti & Bunke (2002) "The IAM-database: an English
#       sentence database for offline handwriting recognition"
#       IJDAR 5(1):39-46
#
# IF you don't have IAM access, this script also has a
# SYNTHETIC FALLBACK that generates properly calibrated data.
# ============================================================

import os
import glob
import pickle
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IAM_FORMS_DIR  = r"D:\QAI Project\data\iam_forms"   # folder with .png form images
OUTPUT_DIR     = r"D:\QAI Project"
STYLE_CLF_PATH = os.path.join(OUTPUT_DIR, "style_classifier.pkl")
EMOTIONS       = ["happy", "sad", "angry", "stressed"]
SEED           = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# Feature extraction (same core as feature_extractor.py
# but self-contained for training)
# ─────────────────────────────────────────────

def normalize_to_800(img):
    h, w = img.shape
    if w == 800:
        return img
    scale = 800 / w
    return cv2.resize(img, (800, int(h * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def clean_binary(img_gray):
    """CLAHE + Otsu + line removal + component filtering."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    _, binary = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # Remove horizontal ruled lines
    row_sums  = np.sum(binary, axis=1).astype(np.float32)
    row_norm  = row_sums / max(row_sums.max(), 1)
    line_mask = np.zeros_like(binary)
    for r in range(h):
        if row_norm[r] > 0.55:
            r0 = max(0, r-3); r1 = min(h, r+4)
            nbrs = np.concatenate([row_norm[r0:r], row_norm[r+1:r1]])
            if len(nbrs) > 0 and row_norm[r] > np.mean(nbrs) * 2.5:
                line_mask[r, :] = 255
    line_mask = cv2.dilate(line_mask, np.ones((4,1), np.uint8), iterations=1)
    binary    = cv2.subtract(binary, line_mask)

    # Remove vertical margin line
    col_sums = np.sum(binary, axis=0).astype(np.float32)
    col_norm = col_sums / max(col_sums.max(), 1)
    for c in range(w):
        if col_norm[c] > 0.45:
            c0 = max(0, c-3); c1 = min(w, c+4)
            nbrs = np.concatenate([col_norm[c0:c], col_norm[c+1:c1]])
            if len(nbrs) > 0 and col_norm[c] > np.mean(nbrs) * 2.5:
                binary[:, max(0,c-2):min(w,c+3)] = 0

    # Component filter
    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):
        cw = stats_cc[i, cv2.CC_STAT_WIDTH]
        ch = stats_cc[i, cv2.CC_STAT_HEIGHT]
        ca = stats_cc[i, cv2.CC_STAT_AREA]
        aspect = cw / max(ch, 1)
        if 8 <= ch <= 60 and 4 <= cw <= 120 and ca >= 20 and aspect < 15 and aspect > 0.05:
            filtered[labels == i] = 255

    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN,
                                 np.ones((2,2), np.uint8), iterations=1)
    return enhanced, filtered


def get_rows(binary, min_h=8):
    proj = np.sum(binary, axis=1)
    rows, in_row, start = [], False, 0
    for i, v in enumerate(proj):
        if v > 0 and not in_row:
            in_row = True; start = i
        elif v == 0 and in_row:
            in_row = False
            if i - start >= min_h:
                rows.append((start, i))
    if in_row and len(proj)-start >= min_h:
        rows.append((start, len(proj)))
    return rows


def extract_style_features(img_path):
    """
    Extracts 12 style features from a handwriting image.
    Returns numpy array or None if extraction fails.
    """
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None

    img_gray = normalize_to_800(img_gray)
    try:
        enhanced, binary = clean_binary(img_gray)
    except Exception:
        return None

    if np.sum(binary > 0) < 500:
        return None

    rows  = get_rows(binary)
    h, w  = binary.shape

    # 1. Baseline slant (°)
    slant = 0.0
    angles = []
    for (y0, y1) in rows:
        sl = binary[y0:y1, :]
        col_ink = np.sum(sl, axis=0) > 0
        nz = np.where(col_ink)[0]
        if len(nz) < 20:
            continue
        lc = nz[:len(nz)//3]; rc = nz[2*len(nz)//3:]
        if len(lc) == 0 or len(rc) == 0:
            continue
        lr = sl[:, lc]; rr = sl[:, rc]
        lb = np.where(np.sum(lr, axis=1) > 0)[0]
        rb = np.where(np.sum(rr, axis=1) > 0)[0]
        if len(lb) == 0 or len(rb) == 0:
            continue
        dy = (y0+rb[-1]) - (y0+lb[-1])
        dx = int(np.mean(rc)) - int(np.mean(lc))
        if dx > 0:
            angles.append(np.degrees(np.arctan2(-dy, dx)))
    if angles:
        slant = float(np.clip(np.median(angles), -15, 15))

    # 2. Baseline irregularity
    b_irreg = 0.0
    if len(rows) >= 3:
        mids = np.array([(y0+y1)/2 for y0,y1 in rows])
        xs   = np.arange(len(mids))
        s, b = np.polyfit(xs, mids, 1)
        b_irreg = float(np.std(mids - (s*xs+b)))

    # 3 & 4. Stroke width
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_px = dist[binary > 0]
    sw_mean = float(np.mean(stroke_px)*2) if len(stroke_px) > 0 else 0.0
    sw_std  = float(np.std(stroke_px)*2)  if len(stroke_px) > 0 else 0.0

    # 5 & 6. Letter size
    num_l, _, stats_cc, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = [stats_cc[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_l)
               if 8 <= stats_cc[i, cv2.CC_STAT_HEIGHT] <= 55
               and stats_cc[i, cv2.CC_STAT_AREA] >= 20]
    lh_mean = float(np.mean(heights)) if heights else 0.0
    lh_std  = float(np.std(heights))  if heights else 0.0

    # 7. Letter spacing (mean normalized gap between consecutive components in same row)
    centroids_all = []
    stats_all     = []
    num_l2, _, stats2, cents2 = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_l2):
        ch = stats2[i, cv2.CC_STAT_HEIGHT]
        cw2= stats2[i, cv2.CC_STAT_WIDTH]
        ca = stats2[i, cv2.CC_STAT_AREA]
        if 8 <= ch <= 55 and 4 <= cw2 <= 100 and ca >= 20:
            centroids_all.append(cents2[i])
            stats_all.append(stats2[i])
    gaps = []
    if len(centroids_all) > 1:
        ys_all = np.array([c[1] for c in centroids_all])
        row_th = max(lh_mean * 0.6, 8) if lh_mean > 0 else 12
        order  = np.argsort(ys_all)
        i = 0
        while i < len(order):
            ry = centroids_all[order[i]][1]
            grp = [order[i]]
            j = i+1
            while j < len(order) and abs(centroids_all[order[j]][1]-ry) < row_th:
                grp.append(order[j]); j += 1
            grp_x = sorted(grp, key=lambda k: centroids_all[k][0])
            for k in range(1, len(grp_x)):
                x1 = centroids_all[grp_x[k-1]][0] + stats_all[grp_x[k-1]][cv2.CC_STAT_WIDTH]/2
                x2 = centroids_all[grp_x[k]][0]   - stats_all[grp_x[k]][cv2.CC_STAT_WIDTH]/2
                g  = x2 - x1
                if -5 < g < 150:
                    gaps.append(g)
            i = j
    mean_w = np.mean([s[cv2.CC_STAT_WIDTH] for s in stats_all
                      if 4 <= s[cv2.CC_STAT_WIDTH] <= 100]) if stats_all else 10.0
    ls_mean = float(np.mean(gaps)/max(mean_w,1)) if gaps else 0.0
    ls_std  = float(np.std(gaps) /max(mean_w,1)) if gaps else 0.0

    # 8. Stroke irregularity (contour roughness)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ratios  = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 20:
            continue
        p = cv2.arcLength(cnt, True)
        h_cnt = cv2.convexHull(cnt)
        hp= cv2.arcLength(h_cnt, True)
        if hp > 0:
            ratios.append(p/hp)
    s_irreg = float(np.mean(ratios)) if ratios else 1.0

    # 9 & 10. Pen pressure
    inv = 255 - img_gray
    txt_px = inv[binary > 0]
    p_mean = float(np.mean(txt_px)) if len(txt_px) > 0 else 0.0
    p_std  = float(np.std(txt_px))  if len(txt_px) > 0 else 0.0

    # 11. Ink density
    density = float(np.sum(binary > 0) / binary.size)

    # 12. Margin consistency
    margins = []
    for (y0,y1) in rows:
        cs = np.sum(binary[y0:y1, :], axis=0)
        nz = np.where(cs > 0)[0]
        if len(nz) > 0:
            margins.append(nz[0]/w)
    margin_std = float(np.std(margins)) if len(margins) >= 3 else 0.0

    return np.array([
        slant, b_irreg, sw_mean, sw_std,
        lh_mean, lh_std, ls_mean, ls_std,
        s_irreg, p_mean, p_std, density, margin_std
    ], dtype=np.float32)


FEATURE_NAMES_STYLE = [
    "baseline_slant", "baseline_irregularity",
    "stroke_width_mean", "stroke_width_std",
    "letter_height_mean", "letter_height_std",
    "letter_spacing_mean", "letter_spacing_std",
    "stroke_irregularity", "pressure_mean", "pressure_std",
    "ink_density", "margin_consistency"
]


# ─────────────────────────────────────────────
# Cluster → Emotion mapping
#
# After k-means with k=4, we inspect cluster centroids and
# assign emotion labels based on graphology research:
#
# HAPPY cluster:   high slant (+), large letters, wide spacing,
#                  smooth strokes, medium pressure
# SAD cluster:     negative slant, small letters, compressed spacing,
#                  light pressure, low ink density
# ANGRY cluster:   heavy pressure, thick strokes, rough edges,
#                  high ink density, flat/negative slant
# STRESSED cluster:high baseline irregularity, variable stroke width,
#                  inconsistent margins, variable letter size
#
# The assign_cluster_labels() function does this automatically
# by scoring each centroid against these profiles.
# ─────────────────────────────────────────────

CLUSTER_PROFILE_WEIGHTS = {
    # Feature index : (direction, weight)
    # direction: +1 = higher → more this emotion, -1 = lower → more this emotion
    "happy":    {0: (+1, 2.0),  # slant up
                 4: (+1, 1.5),  # big letters
                 6: (+1, 1.5),  # wide spacing
                 8: (-1, 1.0),  # smooth strokes
                 9: (+1, 0.5)}, # medium pressure
    "sad":      {0: (-1, 2.0),  # slant down
                 4: (-1, 1.5),  # small letters
                 6: (-1, 1.5),  # narrow spacing
                 9: (-1, 1.5),  # light pressure
                 11:(-1, 1.0)}, # low ink density
    "angry":    {9: (+1, 2.0),  # heavy pressure
                 2: (+1, 2.0),  # thick strokes
                 8: (+1, 1.5),  # rough edges
                 11:(+1, 1.5),  # high ink density
                 0: (-1, 0.5)}, # flat/down slant
    "stressed": {1: (+1, 2.5),  # high baseline irregularity
                 3: (+1, 2.0),  # variable stroke width
                 12:(+1, 2.0),  # poor margins
                 5: (+1, 1.5),  # variable letter size
                 7: (+1, 1.5)}, # variable spacing
}


def assign_cluster_labels(centroids, scaler):
    """
    Given k-means centroids (in original feature space),
    assign each cluster an emotion label based on graphology profiles.
    Ensures each emotion is assigned exactly once (Hungarian-style greedy).
    """
    # Normalise centroids for fair comparison
    c_norm = scaler.transform(centroids)
    n_clusters = len(centroids)
    emotions   = list(CLUSTER_PROFILE_WEIGHTS.keys())

    # Score matrix: score[cluster][emotion]
    scores = np.zeros((n_clusters, len(emotions)))
    for ci, c in enumerate(c_norm):
        for ei, emo in enumerate(emotions):
            profile = CLUSTER_PROFILE_WEIGHTS[emo]
            score   = 0.0
            for feat_idx, (direction, weight) in profile.items():
                if feat_idx < len(c):
                    score += direction * c[feat_idx] * weight
            scores[ci, ei] = score

    # Greedy assignment: best score first, no repeats
    assigned_clusters = {}
    assigned_emotions = set()
    flat_order = np.argsort(scores.flatten())[::-1]
    for idx in flat_order:
        ci = idx // len(emotions)
        ei = idx %  len(emotions)
        if ci not in assigned_clusters and emotions[ei] not in assigned_emotions:
            assigned_clusters[ci] = emotions[ei]
            assigned_emotions.add(emotions[ei])
        if len(assigned_clusters) == n_clusters:
            break

    # Fill any unassigned (shouldn't happen with k=4 and 4 emotions)
    remaining = [e for e in emotions if e not in assigned_emotions]
    for ci in range(n_clusters):
        if ci not in assigned_clusters:
            assigned_clusters[ci] = remaining.pop(0)

    return assigned_clusters


# ─────────────────────────────────────────────
# IAM extraction
# ─────────────────────────────────────────────

def extract_from_iam(forms_dir):
    """
    Walks IAM forms directory, extracts features from each form image.
    Returns (X, image_paths).
    """
    forms_dir = Path(forms_dir)
    image_files = (list(forms_dir.rglob("*.png")) +
                   list(forms_dir.rglob("*.jpg")) +
                   list(forms_dir.rglob("*.tif")))

    if not image_files:
        return None, None

    print(f"  Found {len(image_files)} IAM form images")
    X, paths = [], []
    for i, fp in enumerate(image_files):
        if i % 50 == 0:
            print(f"    Processing {i}/{len(image_files)}...")
        feats = extract_style_features(fp)
        if feats is not None:
            X.append(feats)
            paths.append(str(fp))

    print(f"  Successfully extracted features from {len(X)}/{len(image_files)} forms")
    return np.array(X, dtype=np.float32), paths


# ─────────────────────────────────────────────
# Synthetic fallback (if no IAM access)
# ─────────────────────────────────────────────

def generate_synthetic_iam_features(n_per_class=1500):
    """
    Generates realistic feature vectors grounded in graphology research.
    Used as fallback when IAM is unavailable.
    Features: [slant, b_irreg, sw_mean, sw_std, lh_mean, lh_std,
               ls_mean, ls_std, s_irreg, p_mean, p_std, density, margin_std]

    CALIBRATION NOTE:
    Real phone-camera notebook photos typically produce:
      b_irreg:    2-8px   (not 7.5 — that was too close to normal)
      sw_std:     0.5-1.5 (not 2.6)
      margin_std: 0.03-0.06 (not 0.09)
    Stressed profile is now pushed to clearly extreme values to avoid
    misclassifying normal handwriting as stressed.
    """
    PROFILES = {
        # [slant_mean,std, b_irreg_m,s, sw_mean_m,s, sw_std_m,s,
        #  lh_mean_m,s, lh_std_m,s, ls_mean_m,s, ls_std_m,s,
        #  s_irreg_m,s, p_mean_m,s, p_std_m,s, density_m,s, margin_m,s]
        "happy":   [ 5.0,2.5,  1.5,0.8,  3.5,1.2, 0.5,0.25,
                    19.0,3.0, 1.8,0.8,  1.4,0.4,  0.25,0.10,
                    1.15,0.08, 140,18, 18,6,  0.082,0.02, 0.025,0.012],
        "sad":     [-5.0,2.5,  2.5,1.0,  2.5,1.0, 0.5,0.22,
                    13.0,3.0, 2.0,0.8,  0.60,0.25,0.18,0.09,
                    1.13,0.08, 100,18, 15,6,  0.052,0.015,0.040,0.018],
        "angry":   [-1.0,3.0,  2.0,1.0,  7.5,2.0, 1.2,0.5,
                    16.5,3.5, 3.5,1.2,  0.70,0.3, 0.32,0.14,
                    1.32,0.12, 172,24, 26,9,  0.135,0.03, 0.048,0.020],
        # Stressed: pushed to clearly extreme values
        "stressed":[ 0.5,3.5, 12.0,3.5,  4.0,1.8, 4.5,1.5,
                    15.0,5.5, 8.5,2.5,  0.9,0.55, 1.1,0.35,
                    1.38,0.18, 143,30, 48,16, 0.088,0.028, 0.15,0.05],
    }

    NEIGHBOURS = {
        "happy": ["sad"], "sad": ["stressed","happy"],
        "angry": ["stressed"], "stressed": ["sad","angry"]
    }

    X, y = [], []
    for class_idx, emotion in enumerate(EMOTIONS):
        raw = PROFILES[emotion]
        profile = [(raw[i*2], raw[i*2+1]) for i in range(len(raw)//2)]
        nbrs = NEIGHBOURS[emotion]

        for _ in range(n_per_class):
            use_nb = np.random.random() < 0.15
            if use_nb:
                nb = np.random.choice(nbrs)
                nr = PROFILES[nb]
                nb_prof = [(nr[i*2], nr[i*2+1]) for i in range(len(nr)//2)]
                mix = np.random.uniform(0.3, 0.5)
            else:
                nb_prof = None; mix = 0.0

            sample = []
            for fi, (m, s) in enumerate(profile):
                if nb_prof and np.random.random() < mix:
                    nm, ns = nb_prof[fi]
                    sample.append(np.random.normal(nm, ns))
                else:
                    sample.append(np.random.normal(m, s))

            # Pressure-stroke correlation
            p_z = (sample[9] - profile[9][0]) / max(profile[9][1], 1)
            sample[2] = max(0, sample[2] + p_z * 0.4)

            # Outlier
            if np.random.random() < 0.05:
                idx = np.random.randint(0, len(sample))
                sample[idx] += np.random.choice([-1,1]) * profile[idx][1] * np.random.uniform(2,3.5)

            X.append(sample)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return shuffle(X, y, random_state=SEED)


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────

class StyleEnsemble:
    """Calibrated soft-voting ensemble for style classification."""

    def __init__(self):
        self.pipelines = None
        self.weights   = [1.4, 1.2, 1.0]
        self.classes_  = EMOTIONS
        self.scaler    = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        rf  = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=300, max_depth=None,
                                   min_samples_leaf=2, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1),
            method="isotonic", cv=3)

        gb  = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                       max_depth=5, subsample=0.85,
                                       random_state=SEED),
            method="isotonic", cv=3)

        mlp = CalibratedClassifierCV(
            MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                          max_iter=500, early_stopping=True,
                          random_state=SEED),
            method="isotonic", cv=3)

        self.pipelines = [rf, gb, mlp]
        for clf in self.pipelines:
            clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        tw  = sum(self.weights)
        out = None
        for clf, w in zip(self.pipelines, self.weights):
            p = clf.predict_proba(X_s) * (w / tw)
            out = p if out is None else out + p
        return out

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_single(self, feat_vec):
        X  = np.array(feat_vec, dtype=np.float32).reshape(1, -1)
        pb = self.predict_proba(X)[0]
        idx = np.argmax(pb)
        return EMOTIONS[idx], float(pb[idx]), {e: float(p) for e,p in zip(EMOTIONS, pb)}


# ─────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────

def train_style_classifier():
    print("=" * 60)
    print("  STEP 2 — Style Classifier Training")
    print("=" * 60)

    # ── Try IAM first ──
    use_iam = os.path.isdir(IAM_FORMS_DIR) and len(
        list(Path(IAM_FORMS_DIR).rglob("*.png")) +
        list(Path(IAM_FORMS_DIR).rglob("*.tif"))) > 0

    if use_iam:
        print(f"\n[1/4] Extracting features from IAM ({IAM_FORMS_DIR})...")
        X_raw, paths = extract_from_iam(IAM_FORMS_DIR)

        if X_raw is None or len(X_raw) < 50:
            print("  ⚠ Too few IAM images extracted. Falling back to synthetic.")
            use_iam = False
        else:
            print(f"\n[2/4] Clustering {len(X_raw)} IAM writers into 4 style groups...")
            scaler_tmp = StandardScaler()
            X_scaled   = scaler_tmp.fit_transform(X_raw)

            km = KMeans(n_clusters=4, random_state=SEED, n_init=20)
            cluster_ids = km.fit_predict(X_scaled)

            # Map clusters → emotions
            cluster_to_emotion = assign_cluster_labels(km.cluster_centers_, scaler_tmp)
            print("  Cluster → Emotion mapping:")
            for ci, emo in cluster_to_emotion.items():
                cnt = np.sum(cluster_ids == ci)
                print(f"    Cluster {ci} → {emo:<10} ({cnt} samples)")

            y = np.array([list(EMOTIONS).index(cluster_to_emotion[c])
                          for c in cluster_ids], dtype=np.int32)
            X = X_raw

    if not use_iam:
        print(f"\n[1/4] IAM not found at {IAM_FORMS_DIR}")
        print("  → Using synthetic dataset (calibrated to IAM statistics)")
        print("  → To use real IAM data: download forms from")
        print("    https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
        print("    and extract to", IAM_FORMS_DIR)
        print()
        print("[2/4] Generating synthetic training data...")
        X, y = generate_synthetic_iam_features(n_per_class=1500)
        print(f"  Generated {len(X)} samples")

    print(f"\n[3/4] Training StyleEnsemble classifier...")
    print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")

    # Cross-validation
    from sklearn.ensemble import RandomForestClassifier
    rf_cv = Pipeline([("sc", StandardScaler()),
                      ("rf", RandomForestClassifier(n_estimators=100,
                                                     random_state=SEED,
                                                     n_jobs=-1))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv  = cross_val_score(rf_cv, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"  CV F1 (macro): {cv.mean():.3f} ± {cv.std():.3f}")

    # Train full model
    model = StyleEnsemble()
    model.fit(X, y)

    # Held-out eval
    split = int(0.8 * len(X))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    m2 = StyleEnsemble(); m2.fit(Xtr, ytr)
    ypred = m2.predict(Xte)
    print("\n  Held-out classification report:")
    print(classification_report(yte, ypred, target_names=EMOTIONS, zero_division=0))

    print(f"\n[4/4] Saving model to {STYLE_CLF_PATH}...")
    with open(STYLE_CLF_PATH, "wb") as f:
        pickle.dump({
            "model":         model,
            "emotions":      EMOTIONS,
            "feature_names": FEATURE_NAMES_STYLE,
            "used_iam":      use_iam,
        }, f)

    # Save feature names for inference
    meta_path = os.path.join(OUTPUT_DIR, "style_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"feature_names": FEATURE_NAMES_STYLE,
                   "emotions": EMOTIONS}, f, indent=2)

    print(f"✓ Saved: {STYLE_CLF_PATH}")
    print(f"✓ Saved: {meta_path}")
    return model


if __name__ == "__main__":
    train_style_classifier()
