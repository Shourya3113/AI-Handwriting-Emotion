# emnist_emotion_full.py
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ROOT = r"D:\QAI Project"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "emnist_emotion_dataset")

EMOTIONS = ["happy", "sad", "angry", "stressed"]
TARGET_PER_EMOTION = 5000  # reduce for quick testing
IMAGE_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

# ----------------------------
# Emotion augmentation
# ----------------------------
def apply_emotion_effect(img, emotion):
    img = img.copy()
    if img.mean() < 127:
        img = 255 - img

    if emotion == 'happy':
        M = cv2.getRotationMatrix2D((IMAGE_SIZE/2, IMAGE_SIZE/2), 5, 1.08)
        out = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), borderValue=255)
        out = cv2.GaussianBlur(out, (3,3), 0.5)
        out = cv2.convertScaleAbs(out, alpha=1.1, beta=10)
    elif emotion == 'sad':
        M = cv2.getRotationMatrix2D((IMAGE_SIZE/2, IMAGE_SIZE/2), -4, 0.95)
        out = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), borderValue=255)
        out = cv2.GaussianBlur(out, (5,5), 1.2)
        out = cv2.convertScaleAbs(out, alpha=0.85, beta=-10)
    elif emotion == 'angry':
        inv = 255 - img
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(inv, kernel, iterations=2)
        thick = 255 - dilated
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        out = np.clip(thick.astype(np.int16) - noise, 0, 255).astype(np.uint8)
    elif emotion == 'stressed':
        rows, cols = img.shape
        out = np.ones_like(img) * 255
        amplitude = 2.2
        period = 16.0 + random.random()*8.0
        for r in range(rows):
            offset = int(amplitude * np.sin(2 * np.pi * r / period) + random.randint(-1,1))
            row = np.roll(img[r], offset)
            out[r] = row
        noise = np.random.normal(0, 6, img.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        out = img
    return out.astype(np.uint8)

# ----------------------------
# Generate emotion dataset
# ----------------------------
def generate_emotion_dataset_from_tensor(images_tensor, out_dir=OUTPUT_DIR, per_emotion=TARGET_PER_EMOTION):
    os.makedirs(out_dir, exist_ok=True)
    N = len(images_tensor)
    rng = np.random.default_rng()

    for emo in EMOTIONS:
        emo_dir = Path(out_dir) / emo
        emo_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating {per_emotion} images for '{emo}'...")
        inds = rng.integers(0, N, per_emotion)
        for i, idx in enumerate(tqdm(inds)):
            img_tensor = images_tensor[idx]
            img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            aug = apply_emotion_effect(img_np, emo)
            im = Image.fromarray(aug)
            im.save(emo_dir / f"{emo}_{i:06d}.png")

# ----------------------------
# CNN
# ----------------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Train & evaluation
# ----------------------------
def train_and_eval(dataset_dir):
    tf_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=8, translate=(0.05,0.05), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tf_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_ds = datasets.ImageFolder(dataset_dir, transform=tf_train)
    class_names = full_ds.classes
    print("Classes:", class_names)

    total = len(full_ds)
    train_len = int(0.7 * total)
    val_len = int(0.2 * total)
    test_len = total - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    val_ds.dataset.transform = tf_eval
    test_ds.dataset.transform = tf_eval

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = EmotionCNN(num_classes=len(class_names)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    best_val = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)
        sched.step()
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = correct / total_val
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "best_emotion_cnn.pth"))
            print("Saved best model.")

    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "best_emotion_cnn.pth")))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("Loading EMNIST from local files...")
    emnist_train = datasets.EMNIST(
        root=DATA_ROOT,
        split="byclass",
        train=True,
        download=True,  # Let torchvision extract your manual zip
        transform=transforms.ToTensor()
    )

    print("Total EMNIST train samples:", len(emnist_train))

    images_tensor = [img for img, _ in emnist_train]

    # Generate emotion-labeled dataset
    generate_emotion_dataset_from_tensor(images_tensor, out_dir=OUTPUT_DIR, per_emotion=TARGET_PER_EMOTION)

    # Train & evaluate CNN
    train_and_eval(OUTPUT_DIR)
