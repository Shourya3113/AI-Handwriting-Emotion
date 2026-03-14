# train_classifier.py
# Generates a psychologically-grounded synthetic dataset and trains an ensemble classifier
# Based on handwriting psychology / graphology research:
#   Extending Handwriting Analysis for Mental Health Detection - Champa & Rani (2010)
#   Graphology: Personality and Emotion from Handwriting - various sources

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

EMOTIONS = ["happy", "sad", "angry", "stressed"]
SAMPLES_PER_CLASS = 2000
RANDOM_STATE = 42
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_classifier.pkl")

np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────
# Psychologically-grounded feature distributions per emotion
# Each tuple: (mean, std) for each of the 16 features in order
# Features: baseline_slant, baseline_irregularity, stroke_width_mean,
#           stroke_width_std, letter_height_mean, letter_height_std,
#           letter_width_mean, letter_width_std, letter_spacing_mean,
#           letter_spacing_std, word_spacing, stroke_irregularity,
#           pressure_mean, pressure_std, margin_consistency, ink_density
# ─────────────────────────────────────────────────────────────────

EMOTION_PROFILES = {

    # HAPPY: Upward slant, larger letters, wide spacing, smooth strokes,
    #        consistent margins, medium pressure, light/airy feel
    # Sources: Rising baseline = optimism; large letters = extroversion;
    #          wide spacing = generosity / openness
    # NOTE: std values are intentionally wide to model real human variation
    "happy": [
        ( 5.0,  3.5),   # baseline_slant: upward (+5°), but humans vary a lot
        ( 2.5,  1.8),   # baseline_irregularity: low-medium
        ( 4.5,  1.5),   # stroke_width_mean: medium
        ( 0.9,  0.5),   # stroke_width_std
        (18.0,  4.5),   # letter_height_mean: larger letters
        ( 3.0,  1.5),   # letter_height_std
        (14.0,  3.5),   # letter_width_mean: wider
        ( 2.5,  1.2),   # letter_width_std
        ( 1.3,  0.5),   # letter_spacing_mean: wide
        ( 0.35, 0.18),  # letter_spacing_std
        ( 0.07, 0.03),  # word_spacing: wide
        ( 1.18, 0.12),  # stroke_irregularity: smooth
        (138.0, 25.0),  # pressure_mean: medium
        ( 22.0, 10.0),  # pressure_std
        ( 0.04, 0.025), # margin_consistency
        ( 0.08, 0.03),  # ink_density
    ],

    # SAD: Downward slant, small compressed letters, narrow spacing,
    #      light pressure, irregular/drooping baseline, cramped writing
    "sad": [
        (-4.5,  3.5),   # baseline_slant: downward
        ( 4.0,  2.0),   # baseline_irregularity: medium-high (drooping)
        ( 3.0,  1.2),   # stroke_width_mean: thin
        ( 0.7,  0.4),   # stroke_width_std
        (13.0,  3.5),   # letter_height_mean: small
        ( 2.8,  1.2),   # letter_height_std
        ( 9.5,  3.0),   # letter_width_mean: narrow
        ( 2.0,  1.0),   # letter_width_std
        ( 0.65, 0.3),   # letter_spacing_mean: compressed
        ( 0.25, 0.15),  # letter_spacing_std
        ( 0.035,0.02),  # word_spacing: narrow
        ( 1.14, 0.10),  # stroke_irregularity: fairly smooth
        (102.0, 22.0),  # pressure_mean: light
        ( 20.0,  8.0),  # pressure_std
        ( 0.05, 0.025), # margin_consistency: slightly inconsistent
        ( 0.05, 0.015), # ink_density: low (light ink)
    ],

    # ANGRY: Heavy pressure, thick strokes, sharp/angular strokes,
    #        compressed spacing, flat or slightly downward baseline,
    #        high ink density, rough/jagged edges
    "angry": [
        (-1.0,  4.0),   # baseline_slant: flat to slight downward, wide variance
        ( 3.0,  2.0),   # baseline_irregularity: medium
        ( 7.0,  2.0),   # stroke_width_mean: thick (heavy pressure)
        ( 1.5,  0.8),   # stroke_width_std
        (16.0,  4.0),   # letter_height_mean: medium-large
        ( 4.5,  1.8),   # letter_height_std: variable (rushed)
        (11.0,  3.5),   # letter_width_mean
        ( 3.5,  1.5),   # letter_width_std
        ( 0.75, 0.4),   # letter_spacing_mean: compressed
        ( 0.4,  0.2),   # letter_spacing_std
        ( 0.045,0.025), # word_spacing: compressed
        ( 1.32, 0.16),  # stroke_irregularity: jagged
        (170.0, 30.0),  # pressure_mean: very heavy
        ( 30.0, 12.0),  # pressure_std
        ( 0.06, 0.03),  # margin_consistency
        ( 0.13, 0.04),  # ink_density: high
    ],

    # STRESSED: Highly irregular baseline, variable stroke width,
    #           inconsistent letter size and spacing, shaky strokes,
    #           poor margin control, erratic pressure
    "stressed": [
        ( 0.5,  5.0),   # baseline_slant: very variable
        ( 7.5,  3.0),   # baseline_irregularity: very high
        ( 4.2,  2.0),   # stroke_width_mean: medium but variable
        ( 2.8,  1.2),   # stroke_width_std: highly variable
        (15.0,  5.0),   # letter_height_mean
        ( 6.0,  2.0),   # letter_height_std: very variable
        (11.0,  4.5),   # letter_width_mean
        ( 5.0,  1.8),   # letter_width_std: very variable
        ( 0.9,  0.55),  # letter_spacing_mean: erratic
        ( 0.65, 0.25),  # letter_spacing_std: very high
        ( 0.055,0.03),  # word_spacing: erratic
        ( 1.27, 0.18),  # stroke_irregularity: rough/shaky
        (145.0, 35.0),  # pressure_mean: erratic
        ( 45.0, 15.0),  # pressure_std: very high
        ( 0.10, 0.04),  # margin_consistency: very inconsistent
        ( 0.09, 0.03),  # ink_density
    ],
}


def generate_synthetic_dataset(n_per_class=SAMPLES_PER_CLASS):
    """
    Generates realistic synthetic feature vectors.

    Key realism mechanisms:
    1. Wide per-feature std (already in profiles)
    2. Cross-class contamination: 15% of each class borrows features from
       a neighbouring emotion (sad↔stressed, angry↔stressed overlap)
    3. Correlated noise: pressure/stroke_width co-vary within a sample
    4. Hard outliers (5%): one feature takes an extreme value
    """
    # Neighbour map: emotions that genuinely overlap
    NEIGHBOURS = {
        "happy":    ["sad"],               # occasional low-energy happy
        "sad":      ["stressed", "happy"], # sad-stressed boundary is fuzzy
        "angry":    ["stressed"],          # angry-stressed share irregularity
        "stressed": ["sad", "angry"],      # stressed borrows from both
    }

    X, y = [], []
    for class_idx, emotion in enumerate(EMOTIONS):
        profile = EMOTION_PROFILES[emotion]
        neighbours = NEIGHBOURS[emotion]

        for _ in range(n_per_class):
            sample = []

            # 15% chance: mix in one neighbour's distribution for 30-50% of features
            use_neighbour = np.random.random() < 0.15
            if use_neighbour:
                nb_name = np.random.choice(neighbours)
                nb_profile = EMOTION_PROFILES[nb_name]
                mix_ratio = np.random.uniform(0.3, 0.5)
            else:
                nb_profile = None
                mix_ratio = 0.0

            for feat_idx, (mean, std) in enumerate(profile):
                if nb_profile is not None and np.random.random() < mix_ratio:
                    nb_mean, nb_std = nb_profile[feat_idx]
                    val = np.random.normal(nb_mean, nb_std)
                else:
                    val = np.random.normal(mean, std)
                sample.append(val)

            # Correlated noise: pressure and stroke_width should co-vary
            # feat indices: stroke_width_mean=2, pressure_mean=12
            pressure_z = (sample[12] - EMOTION_PROFILES[emotion][12][0]) / max(EMOTION_PROFILES[emotion][12][1], 1)
            sample[2] += pressure_z * 0.4   # heavier pressure → thicker strokes

            # Hard outlier: 5% chance one random feature is extreme
            if np.random.random() < 0.05:
                idx = np.random.randint(0, len(sample))
                _, std = profile[idx]
                sample[idx] += np.random.choice([-1, 1]) * std * np.random.uniform(2.0, 3.5)

            X.append(sample)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return shuffle(X, y, random_state=RANDOM_STATE)


def build_ensemble():
    """
    Builds a soft-voting ensemble of 3 classifiers.
    Each wrapped in a StandardScaler pipeline.
    """
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=RANDOM_STATE
        ))
    ])

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE
        ))
    ])

    return rf, gb, mlp


class EnsembleClassifier:
    """Soft-voting ensemble with probability averaging."""

    def __init__(self, classifiers, weights=None):
        self.classifiers = classifiers
        self.weights = weights if weights else [1.0] * len(classifiers)
        self.classes_ = EMOTIONS

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)
        return self

    def predict_proba(self, X):
        proba_sum = None
        total_weight = sum(self.weights)
        for clf, w in zip(self.classifiers, self.weights):
            p = clf.predict_proba(X) * (w / total_weight)
            proba_sum = p if proba_sum is None else proba_sum + p
        return proba_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_single(self, feature_vector):
        """
        Takes a 1D numpy array of features.
        Returns (emotion_str, confidence, all_probs_dict)
        """
        X = feature_vector.reshape(1, -1)
        proba = self.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        emotion = EMOTIONS[pred_idx]
        confidence = float(proba[pred_idx])
        all_probs = {e: float(p) for e, p in zip(EMOTIONS, proba)}
        return emotion, confidence, all_probs


def train_and_save(model_path=MODEL_PATH):
    print("=" * 55)
    print("  Handwriting Emotion Classifier — Training")
    print("=" * 55)

    print(f"\n[1/4] Generating synthetic dataset ({SAMPLES_PER_CLASS} samples/class)...")
    X, y = generate_synthetic_dataset(SAMPLES_PER_CLASS)
    print(f"      Total samples: {len(X)} | Features: {X.shape[1]}")
    print(f"      Class distribution: {dict(zip(EMOTIONS, np.bincount(y)))}")

    print("\n[2/4] Building ensemble classifiers...")
    rf, gb, mlp = build_ensemble()
    # RF gets highest weight (most robust), MLP slightly lower
    ensemble = EnsembleClassifier([rf, gb, mlp], weights=[1.4, 1.2, 1.0])

    print("\n[3/4] Cross-validation (5-fold stratified)...")
    from sklearn.base import clone

    # CV on RF alone first for quick estimate
    rf_cv = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf_cv, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"      RF CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n[4/4] Training full ensemble on all data...")
    ensemble.fit(X, y)

    # Final evaluation on held-out 20%
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    ensemble_eval = EnsembleClassifier(build_ensemble(), weights=[1.4, 1.2, 1.0])
    ensemble_eval.fit(X_train, y_train)
    y_pred = ensemble_eval.predict(X_test)

    print("\n  Classification Report (held-out 20%):")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the full-data trained ensemble
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(ensemble, f)
    print(f"\n✓ Model saved to: {model_path}")
    return ensemble


def load_model(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train_and_save()
