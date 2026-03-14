# ============================================================
# STEP 1 — Fine-tune DistilBERT on GoEmotions
# Run this on Google Colab (GPU runtime) or local GPU
#
# What this produces:
#   emotion_text_model/   — saved HuggingFace model
#   emotion_text_model/label_map.json
#
# Cite: Demszky et al. (2020) "GoEmotions: A Dataset of
#       Fine-Grained Emotions" — ACL 2020
# ============================================================

# ── Install (Colab only, comment out if local) ──
# !pip install transformers datasets torch scikit-learn -q

import json
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report, f1_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR   = "emotion_text_model"
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 4
LR           = 2e-5
SEED         = 42

# GoEmotions has 28 emotion labels.
# We map them to our 4 classes + neutral (which we drop at inference).
#
# Mapping rationale (Demszky et al. Table 1):
GOEMOTION_TO_4 = {
    # HAPPY
    "admiration":    "happy",
    "amusement":     "happy",
    "excitement":    "happy",
    "joy":           "happy",
    "love":          "happy",
    "optimism":      "happy",
    "pride":         "happy",
    "relief":        "happy",
    "gratitude":     "happy",
    "approval":      "happy",
    # SAD
    "disappointment":"sad",
    "embarrassment": "sad",
    "grief":         "sad",
    "remorse":       "sad",
    "sadness":       "sad",
    # ANGRY
    "anger":         "angry",
    "annoyance":     "angry",
    "disgust":       "angry",
    # STRESSED
    "confusion":     "stressed",
    "fear":          "stressed",
    "nervousness":   "stressed",
    "surprise":      "stressed",   # negative surprise
    # NEUTRAL / DROP
    "caring":        "neutral",
    "curiosity":     "neutral",
    "desire":        "neutral",
    "neutral":       "neutral",
    "realization":   "neutral",
    "disapproval":   "neutral",
}

LABELS     = ["happy", "sad", "angry", "stressed", "neutral"]
LABEL2ID   = {l: i for i, l in enumerate(LABELS)}
ID2LABEL   = {i: l for i, l in enumerate(LABELS)}

torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Load + map GoEmotions
# ─────────────────────────────────────────────
def load_and_map():
    print("[1/5] Loading GoEmotions dataset...")
    ds = load_dataset("go_emotions", "simplified")

    # GoEmotions label names
    go_label_names = ds["train"].features["labels"].feature.names

    def map_example(example):
        # Multi-label: take the first label that maps to one of our 4
        mapped = "neutral"
        for lid in example["labels"]:
            go_name = go_label_names[lid]
            candidate = GOEMOTION_TO_4.get(go_name, "neutral")
            if candidate != "neutral":
                mapped = candidate
                break
            mapped = "neutral"
        example["emotion"] = LABEL2ID[mapped]
        return example

    ds = ds.map(map_example)
    print(f"    Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")

    # Show class distribution
    from collections import Counter
    counts = Counter(ds["train"]["emotion"])
    print("    Train class distribution:")
    for lid, cnt in sorted(counts.items()):
        print(f"      {ID2LABEL[lid]:<12} {cnt:>6}")
    return ds


# ─────────────────────────────────────────────
# Tokenize
# ─────────────────────────────────────────────
def tokenize_dataset(ds, tokenizer):
    print("[2/5] Tokenizing...")
    def tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
    ds = ds.map(tok, batched=True)
    # GoEmotions already has a 'labels' column — drop it first, then rename our 'emotion' column
    ds = ds.remove_columns("labels")
    ds = ds.rename_column("emotion", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    return {"f1_macro": f1, "accuracy": acc}


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
def train():
    ds = load_and_map()

    print("[3/5] Loading tokenizer + model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    ds = tokenize_dataset(ds, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Class weights to handle imbalance (happy >> other classes in GoEmotions)
    from collections import Counter
    # Get label counts before torch format (access as plain python list)
    counts = Counter(ds["train"].with_format(None)["labels"])
    total  = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABELS) * counts[i]) for i in range(len(LABELS))],
        dtype=torch.float
    )
    print(f"    Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    # Custom Trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        seed=SEED,
        report_to="none",
    )

    print("[4/5] Training...")
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("[5/5] Evaluating on test set...")
    preds_out  = trainer.predict(ds["test"])
    preds      = np.argmax(preds_out.predictions, axis=1)
    labels_out = preds_out.label_ids
    print(classification_report(labels_out, preds,
                                  target_names=LABELS, zero_division=0))

    # Save model + label map
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    print(f"\n✓ Model saved to ./{OUTPUT_DIR}/")
    print("  → Copy the entire 'emotion_text_model/' folder to D:\\QAI Project\\")


if __name__ == "__main__":
    train()