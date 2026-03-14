# ===============================================================
# Quantum AI Handwriting Emotion Classifier (Laptop-Optimized)
# ===============================================================

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

from qiskit_aer.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from scipy.optimize import minimize

# ===============================================================
# STEP 1: Load images from folders
# ===============================================================
DATA_ROOT = r"D:\QAI Project\emnist_emotion_dataset"
emotions = ["angry", "happy", "sad", "stressed"]

X = []
y = []

for label in emotions:
    folder = os.path.join(DATA_ROOT, label)
    for file in os.listdir(folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).convert('L')  # grayscale
            img = img.resize((28, 28))
            img_array = np.array(img).flatten()
            X.append(img_array)
            y.append(label)

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)
print(f"✅ Loaded {X.shape[0]} images with {X.shape[1]} features each")

# ===============================================================
# STEP 2: Encode labels
# ===============================================================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
labels = encoder.classes_

# ===============================================================
# STEP 3: Train-test split & scaling
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================================
# STEP 4: PCA for quantum feature reduction (auto)
# ===============================================================
pca_full = PCA()
pca_full.fit(X_train_scaled)
explained_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)

# Automatically pick min qubits to retain 90-95% variance
num_qubits = np.searchsorted(explained_variance_ratio, 0.95) + 1
if num_qubits > 8:
    num_qubits = 8  # cap to 8 qubits for CPU speed
print(f"✅ Selected {num_qubits} PCA components/qubits (~95% variance)")

pca = PCA(n_components=num_qubits)
X_train_q = pca.fit_transform(X_train_scaled)
X_test_q = pca.transform(X_test_scaled)

# ===============================================================
# STEP 5: Reduce training set size for speed
# ===============================================================
MAX_TRAIN_SAMPLES = 1000  # faster training
if X_train_q.shape[0] > MAX_TRAIN_SAMPLES:
    indices = np.random.choice(X_train_q.shape[0], MAX_TRAIN_SAMPLES, replace=False)
    X_train_q = X_train_q[indices]
    y_train = y_train[indices]
    print(f"⚡ Using {MAX_TRAIN_SAMPLES} random training samples for faster training")

# ===============================================================
# STEP 6: Define Quantum Circuits (shallow for speed)
# ===============================================================
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)
sampler = Sampler()

# ===============================================================
# STEP 7: Fixed SciPy optimizer wrapper for VQC
# ===============================================================
class ScipyOptimizer:
    def __init__(self, method="SLSQP", options=None):
        self.method = method
        self.options = options if options else {"maxiter": 20}  # lower maxiter for speed

    def minimize(self, fun, x0, **kwargs):
        res = minimize(fun, x0, method=self.method, options=self.options, **kwargs)
        return res

optimizer = ScipyOptimizer(method="SLSQP", options={"maxiter": 20})

# ===============================================================
# STEP 8: Variational Quantum Classifier
# ===============================================================
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    num_qubits=num_qubits,
    sampler=sampler,
    loss=CrossEntropyLoss(),
)

# ===============================================================
# STEP 9: Train
# ===============================================================
print("\n🧠 Training Quantum VQC Model on handwriting images...")
vqc.fit(X_train_q, y_train)

# ===============================================================
# STEP 10: Evaluate
# ===============================================================
predictions = vqc.predict(X_test_q)

print("\n📊 Classification Report:")
print(classification_report(y_test, predictions, target_names=labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
