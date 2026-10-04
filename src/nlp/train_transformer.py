# src/nlp/train_transformer.py
import json
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import classification_report, f1_score
import sys
from collections import Counter
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from nlp.normalize import normalize

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da"   # swap this to compare models
OUTPUT_DIR = ROOT / "models" / "camelbert_intent"

def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            row = json.loads(line)
            texts.append(normalize(row["text"]))
            labels.append(row["intent"])
    return texts, labels

def main():
    train_texts, train_labels = load_jsonl(ROOT / "data" / "processed" / "train.jsonl")
    val_texts, val_labels = load_jsonl(ROOT / "data" / "processed" / "val.jsonl")

    label_list = sorted(set(train_labels) | set(val_labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    train_ds = Dataset.from_dict({
        "text": train_texts,
        "label": [label2id[l] for l in train_labels],
    })
    val_ds = Dataset.from_dict({
        "text": val_texts,
        "label": [label2id[l] for l in val_labels],
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        report_to=[],
    )

     # Compute class weights inversely proportional to frequency
    label_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (len(label_list) * label_counts[l]) for l in label_list],
        dtype=torch.float
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, len(label_list)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids
    print(classification_report(
        [id2label[i] for i in y_true],
        [id2label[i] for i in y_pred],
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved model to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()