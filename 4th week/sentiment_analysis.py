# ============================================================
# Week 4 – Day 1 Assignment
# Sentiment Analysis using RNN, LSTM & GRU
# Dataset: Local IMDB Dataset.csv
# ============================================================

# ---------------------------
# 1. Libraries
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2. Load Local Dataset
# ---------------------------
print("\nLoading local IMDB Dataset.csv ...")
df = pd.read_csv(r"D:\internship-at-DEMP\4th week\IMDB Dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# Convert sentiment to 0/1
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Train-Test Split (80-20)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

print(f"\nTrain size: {len(train_df)}")
print(f"Test size : {len(test_df)}")

# ---------------------------
# 3. Text Cleaning
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)              # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)     # URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)        # only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nCleaning text...")
train_df = train_df.copy()
test_df = test_df.copy()

train_df["review"] = train_df["review"].apply(clean_text)
test_df["review"] = test_df["review"].apply(clean_text)

# ---------------------------
# 4. Tokenization + Vocabulary
# ---------------------------
def tokenize(text):
    return text.split()

print("Building vocabulary...")
counter = Counter()
for text in train_df["review"]:
    counter.update(tokenize(text))

# Keep words with frequency >= 5
min_freq = 5
vocab = {"<PAD>": 0, "<UNK>": 1}

for word, freq in counter.items():
    if freq >= min_freq:
        vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")


def text_to_sequence(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]


def encode_dataset(dataframe, vocab):
    sequences = [text_to_sequence(text, vocab) for text in dataframe["review"]]
    labels = dataframe["label"].tolist()
    return sequences, labels


train_seqs, train_labels = encode_dataset(train_df, vocab)
test_seqs, test_labels = encode_dataset(test_df, vocab)

# ---------------------------
# 5. Custom Dataset + Collate
# ---------------------------
class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded, labels


train_dataset = SentimentDataset(train_seqs, train_labels)
test_dataset = SentimentDataset(test_seqs, test_labels)

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches : {len(test_loader)}")

# ---------------------------
# 6. Model Definitions
# ---------------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        return self.fc(hidden[-1])


# Hyperparameters
VOCAB_SIZE = len(vocab)
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 5

# ---------------------------
# 7. Training & Evaluation Functions
# ---------------------------
def binary_accuracy(preds, y):
    rounded = torch.round(torch.sigmoid(preds))
    correct = (rounded == y).float()
    return correct.mean()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for text, labels in tqdm(loader, desc="Training", leave=False):
        text, labels = text.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, labels in tqdm(loader, desc="Evaluating", leave=False):
            text, labels = text.to(device), labels.to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            preds = torch.round(torch.sigmoid(predictions))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return epoch_loss / len(loader), epoch_acc / len(loader), all_preds, all_labels


def train_model(model, model_name, train_loader, test_loader, epochs=EPOCHS):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_valid_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), f"{model_name.lower()}_best.pth")
            print("  → Best model saved!")

    return history


# ---------------------------
# 8. Train All Three Models
# ---------------------------
models = {
    "SimpleRNN": SimpleRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device),
    "LSTM": LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device),
    "GRU": GRUModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device),
}

histories = {}
for name, model in models.items():
    histories[name] = train_model(model, name, train_loader, test_loader, epochs=EPOCHS)

# ---------------------------
# 9. Final Evaluation + Metrics
# ---------------------------
results = {}

for name, model in models.items():
    model.load_state_dict(torch.load(f"{name.lower()}_best.pth", map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    _, _, preds, labels = evaluate(model, test_loader, criterion)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Confusion Matrix": cm,
    }

    print(f"\n{'=' * 50}")
    print(f"{name} Final Results")
    print(f"{'=' * 50}")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

# ---------------------------
# 10. Comparison Table
# ---------------------------
comparison_df = pd.DataFrame(
    {
        name: {
            "Accuracy": f"{res['Accuracy'] * 100:.2f}%",
            "Precision": f"{res['Precision'] * 100:.2f}%",
            "Recall": f"{res['Recall'] * 100:.2f}%",
            "F1-score": f"{res['F1-score'] * 100:.2f}%",
        }
        for name, res in results.items()
    }
).T

print("\n" + "=" * 60)
print("FINAL COMPARISON TABLE")
print("=" * 60)
print(comparison_df)

comparison_df.to_csv("model_comparison.csv")
print("\nComparison table saved as → model_comparison.csv")

# ---------------------------
# 11. Analysis
# ---------------------------
print("""
============================================================
ANALYSIS
============================================================

1. Which model performed best?
   → Usually LSTM or GRU perform better than Simple RNN.
   → LSTM is strongest at capturing long-term dependencies.

2. Why?
   → Simple RNN suffers from vanishing gradient problem → forgets long context.
   → LSTM has dedicated memory cell + 3 gates → remembers important information longer.
   → GRU is lighter (2 gates) and often almost as good as LSTM.

3. Which model trained the fastest?
   → Simple RNN (least parameters) → then GRU → then LSTM (most parameters).

4. Recommendation for long text sequences:
   → Prefer LSTM or GRU.
   → Very long documents → LSTM is safer.
   → Speed + memory concern → GRU is excellent trade-off.
""")
