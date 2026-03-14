# feature_extractor.py  (v3 — clean, calibrated for 800px)
# Extracts 13 handwriting style features from any input image.
# Handles: notebook paper, lined paper, plain paper, phone photos, scans.

import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

FEATURE_NAMES = [
    "baseline_slant",        # °: upward=happy, downward=sad
    "baseline_irregularity", # px std: high=stressed
    "stroke_width_mean",     # px: thick=angry, thin=sad
    "stroke_width_std",      # px: high=stressed
    "letter_height_mean",    # px: large=happy, small=sad
    "letter_height_std",     # px: high=stressed
    "letter_spacing_mean",   # norm: wide=happy, narrow=sad/angry
    "letter_spacing_std",    # norm: high=stressed
    "stroke_irregularity",   # ratio: rough=angry/stressed
    "pressure_mean",         # 0-255: heavy=angry
    "pressure_std",          # 0-255: high=stressed
    "ink_density",           # 0-1: dense=angry
    "margin_consistency",    # std: high=stressed
]


def load_image(source):
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, Image.Image):
        img = np.array(source.convert("L"))
    else:
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {source}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def normalize_to_800(img):
    h, w = img.shape
    if w == 800:
        return img
    scale = 800 / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (800, int(h * scale)), interpolation=interp)


def clean_notebook(img_gray):
    """
    Full cleaning pipeline:
    1. CLAHE contrast
    2. Otsu binarize
    3. Remove horizontal ruled lines (row projection)
    4. Remove vertical margin line (col projection)
    5. Component-size filtering (keep letter-shaped blobs only)
    6. Morphological noise removal
    """
    h, w = img_gray.shape

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(img_gray)

    # 2. Binarize
    _, binary = cv2.threshold(enh, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Remove horizontal lines
    row_sums = np.sum(binary, axis=1).astype(np.float32)
    row_norm = row_sums / max(row_sums.max(), 1)
    lmask = np.zeros_like(binary)
    for r in range(h):
        if row_norm[r] > 0.55:
            r0 = max(0, r-3); r1 = min(h, r+4)
            nbrs = np.concatenate([row_norm[r0:r], row_norm[r+1:r1]])
            if len(nbrs) > 0 and row_norm[r] > np.mean(nbrs) * 2.5:
                lmask[r, :] = 255
    lmask  = cv2.dilate(lmask, np.ones((4, 1), np.uint8), iterations=1)
    binary = cv2.subtract(binary, lmask)

    # 4. Remove vertical margin line
    col_sums = np.sum(binary, axis=0).astype(np.float32)
    col_norm = col_sums / max(col_sums.max(), 1)
    for c in range(w):
        if col_norm[c] > 0.45:
            c0 = max(0, c-3); c1 = min(w, c+4)
            nbrs = np.concatenate([col_norm[c0:c], col_norm[c+1:c1]])
            if len(nbrs) > 0 and col_norm[c] > np.mean(nbrs) * 2.5:
                binary[:, max(0,c-2):min(w,c+3)] = 0

    # 5. Component filter — keep letter-sized blobs only
    num_l, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, num_l):
        cw = stats_cc[i, cv2.CC_STAT_WIDTH]
        ch = stats_cc[i, cv2.CC_STAT_HEIGHT]
        ca = stats_cc[i, cv2.CC_STAT_AREA]
        asp = cw / max(ch, 1)
        if 8 <= ch <= 60 and 4 <= cw <= 120 and ca >= 20 and 0.05 < asp < 15:
            filtered[labels == i] = 255

    # 6. Noise removal
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN,
                                 np.ones((2,2), np.uint8), iterations=1)
    return enh, filtered


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


def extract_features(source):
    """
    Main entry point. Returns dict of 13 features.
    """
    img = load_image(source)
    img = normalize_to_800(img)
    enh, binary = clean_notebook(img)

    # Fallback if cleaning removed too much
    if np.sum(binary > 0) < 300:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enh2 = clahe.apply(img)
        _, binary = cv2.threshold(enh2, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rows = get_rows(binary)
    h, w = binary.shape

    # ── 1. Baseline slant ──
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
        dy = rb[-1] - lb[-1]
        dx = int(np.mean(rc)) - int(np.mean(lc))
        if dx > 0:
            angles.append(np.degrees(np.arctan2(-dy, dx)))
    slant = float(np.clip(np.median(angles), -15, 15)) if angles else 0.0

    # ── 2. Baseline irregularity ──
    b_irreg = 0.0
    if len(rows) >= 3:
        mids = np.array([(y0+y1)/2 for y0,y1 in rows])
        xs   = np.arange(len(mids))
        s, b = np.polyfit(xs, mids, 1)
        b_irreg = float(np.std(mids - (s*xs + b)))

    # ── 3 & 4. Stroke width ──
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    sp   = dist[binary > 0]
    sw_mean = float(np.mean(sp) * 2) if len(sp) > 0 else 0.0
    sw_std  = float(np.std(sp)  * 2) if len(sp) > 0 else 0.0

    # ── 5 & 6. Letter size ──
    num_l, _, sc2, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = [sc2[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_l)
               if 8 <= sc2[i, cv2.CC_STAT_HEIGHT] <= 55
               and sc2[i, cv2.CC_STAT_AREA] >= 20]
    lh_mean = float(np.mean(heights)) if heights else 0.0
    lh_std  = float(np.std(heights))  if heights else 0.0

    # ── 7 & 8. Letter spacing ──
    num_l2, _, sc3, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    good_idx = [i for i in range(1, num_l2)
                if 8 <= sc3[i, cv2.CC_STAT_HEIGHT] <= 55
                and 4 <= sc3[i, cv2.CC_STAT_WIDTH]  <= 100
                and sc3[i, cv2.CC_STAT_AREA] >= 20]
    gaps = []
    if len(good_idx) > 1:
        row_th = max(lh_mean * 0.6, 8) if lh_mean > 0 else 12
        ys_g   = np.array([cents[i][1] for i in good_idx])
        order  = np.argsort(ys_g)
        ii = 0
        while ii < len(order):
            ry  = cents[good_idx[order[ii]]][1]
            grp = [order[ii]]
            jj  = ii + 1
            while jj < len(order) and abs(cents[good_idx[order[jj]]][1]-ry) < row_th:
                grp.append(order[jj]); jj += 1
            grp_x = sorted(grp, key=lambda k: cents[good_idx[k]][0])
            for k in range(1, len(grp_x)):
                a = good_idx[grp_x[k-1]]; b_idx = good_idx[grp_x[k]]
                x1 = cents[a][0]     + sc3[a,     cv2.CC_STAT_WIDTH] / 2
                x2 = cents[b_idx][0] - sc3[b_idx, cv2.CC_STAT_WIDTH] / 2
                g  = x2 - x1
                if -5 < g < 150:
                    gaps.append(g)
            ii = jj
    mw = np.mean([sc3[i, cv2.CC_STAT_WIDTH] for i in good_idx
                  if 4 <= sc3[i, cv2.CC_STAT_WIDTH] <= 100]) if good_idx else 10.0
    ls_mean = float(np.mean(gaps) / max(mw, 1)) if gaps else 0.0
    ls_std  = float(np.std(gaps)  / max(mw, 1)) if gaps else 0.0

    # ── 9. Stroke irregularity ──
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ratios  = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 20:
            continue
        p  = cv2.arcLength(cnt, True)
        hp = cv2.arcLength(cv2.convexHull(cnt), True)
        if hp > 0:
            ratios.append(p / hp)
    s_irreg = float(np.mean(ratios)) if ratios else 1.0

    # ── 10 & 11. Pen pressure ──
    inv   = 255 - img
    tx    = inv[binary > 0]
    p_mean = float(np.mean(tx)) if len(tx) > 0 else 0.0
    p_std  = float(np.std(tx))  if len(tx) > 0 else 0.0

    # ── 12. Ink density ──
    density = float(np.sum(binary > 0) / binary.size)

    # ── 13. Margin consistency ──
    margins = []
    for (y0, y1) in rows:
        cs = np.sum(binary[y0:y1, :], axis=0)
        nz = np.where(cs > 0)[0]
        if len(nz) > 0:
            margins.append(nz[0] / w)
    margin_std = float(np.std(margins)) if len(margins) >= 3 else 0.0

    features = {
        "baseline_slant":        slant,
        "baseline_irregularity": b_irreg,
        "stroke_width_mean":     sw_mean,
        "stroke_width_std":      sw_std,
        "letter_height_mean":    lh_mean,
        "letter_height_std":     lh_std,
        "letter_spacing_mean":   ls_mean,
        "letter_spacing_std":    ls_std,
        "stroke_irregularity":   s_irreg,
        "pressure_mean":         p_mean,
        "pressure_std":          p_std,
        "ink_density":           density,
        "margin_consistency":    margin_std,
    }
    return features


def features_to_vector(features):
    return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)


def human_readable(features):
    return {
        "Writing slant":       f"{features['baseline_slant']:+.1f}°",
        "Baseline stability":  f"{features['baseline_irregularity']:.1f}px",
        "Stroke thickness":    f"{features['stroke_width_mean']:.1f}px",
        "Letter size":         f"{features['letter_height_mean']:.1f}px",
        "Letter spacing":      f"{features['letter_spacing_mean']:.2f}×",
        "Stroke smoothness":   f"{features['stroke_irregularity']:.2f}",
        "Pen pressure":        f"{features['pressure_mean']:.0f}/255",
        "Pressure variance":   f"{features['pressure_std']:.1f}",
        "Ink density":         f"{features['ink_density']:.3f}",
        "Margin consistency":  f"{features['margin_consistency']:.3f}",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        f = extract_features(sys.argv[1])
        print("\nExtracted Features:")
        print("-" * 40)
        for k, v in f.items():
            print(f"  {k:<28} {v:>8.4f}")
    else:
        print("Usage: python feature_extractor.py <image_path>")
