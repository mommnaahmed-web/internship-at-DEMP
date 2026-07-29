# Week 4 – Sequential Modeling with RNN, LSTM & GRU

**Sentiment Analysis on IMDb Movie Reviews using PyTorch**  
Deep Learning Internship | Day 1 Assignment

---

## Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Theoretical Background](#theoretical-background)
  - [Sequential Data](#sequential-data)
  - [Recurrent Neural Network (RNN)](#recurrent-neural-network-rnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
- [Dataset](#dataset)
- [Models](#models)
- [Hyperparameters](#hyperparameters)
- [Evaluation Metrics](#evaluation-metrics)
- [Analysis](#analysis)
- [Project Structure](#project-structure)
- [Deliverables](#deliverables)
- [Research Paper Presentation](#research-paper-presentation)
- [Literature Review](#literature-review)
- [Requirements](#requirements)

---

## Overview

This project implements and compares three sequential deep learning models — **Simple RNN**, **LSTM**, and **GRU** — for binary sentiment classification on the IMDb Movie Reviews dataset.

The goal is to understand sequential data modeling, the vanishing gradient problem, and how gated architectures (LSTM/GRU) overcome the limitations of traditional RNNs.

---

## Learning Objectives

- Understand sequential data and sequence modeling
- Explain limitations of feedforward networks for sequential tasks
- Describe the architecture of RNN, LSTM, and GRU
- Understand the vanishing gradient problem and how LSTM/GRU solve it
- Build sequence models using PyTorch
- Compare RNN, LSTM, and GRU on the same dataset

---

## Theoretical Background

### Sequential Data

Data where **order matters** — sentences, speech, time-series, video frames. Feedforward networks cannot handle this because they have no memory.

### Recurrent Neural Network (RNN)

RNN maintains a hidden state that carries information across time steps:

```
h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b)
```

**Limitation:** Vanishing Gradient Problem — gradients shrink across long sequences, so the model forgets early information.

### Long Short-Term Memory (LSTM)

Introduced by Hochreiter & Schmidhuber (1997). Uses a **memory cell** and three gates:

| Gate | Role |
|------|------|
| **Forget Gate** | Decides what to erase from memory |
| **Input Gate** | Decides what new information to write |
| **Output Gate** | Decides what to expose as output |

```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)     ← Forget
i_t = σ(W_i · [h_(t-1), x_t] + b_i)     ← Input
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t         ← Cell update
o_t = σ(W_o · [h_(t-1), x_t] + b_o)     ← Output
h_t = o_t ⊙ tanh(C_t)
```

### Gated Recurrent Unit (GRU)

Simplified LSTM with only **two gates** (Update + Reset). Fewer parameters, faster training, comparable performance.

---

## Dataset

| Property | Details |
|----------|---------|
| Name | IMDb Movie Reviews |
| Size | 50,000 reviews |
| Classes | Positive / Negative |
| Split | 80% Train · 20% Test (stratified) |

**Preprocessing:** Lowercasing → HTML/URL removal → Tokenization → Vocabulary (min_freq=5) → Padding

---

## Models

| Model | Architecture |
|-------|-------------|
| **Simple RNN** | Embedding → RNN (2 layers) → Linear |
| **LSTM** | Embedding → LSTM (2 layers) → Linear |
| **GRU** | Embedding → GRU (2 layers) → Linear |

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dim | 100 |
| Hidden Dim | 128 |
| Layers | 2 |
| Dropout | 0.3 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Epochs | 5 |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Results saved to `model_comparison.csv`.

---

## Analysis

| Question | Answer |
|----------|--------|
| **Best model?** | LSTM / GRU outperform Simple RNN |
| **Why?** | RNN suffers from vanishing gradients. LSTM retains long-term context via gated memory. GRU is lighter with similar performance. |
| **Fastest training?** | Simple RNN → GRU → LSTM |
| **Long sequences?** | Prefer **LSTM** for accuracy, **GRU** for speed |

---

## Project Structure

```
4th week/
├── IMDB Dataset.csv
├── sentiment_analysis.py
├── simplernn_best.pth
├── lstm_best.pth
├── gru_best.pth
├── model_comparison.csv
├── LSTM_Paper_Presentation.pptx
├── Literature_Review_LSTM_GRU.docx
└── README.md
```

---

## Deliverables

| Deliverable | File |
|-------------|------|
| Source Code | `sentiment_analysis.py` |
| Trained Models | `*_best.pth` |
| Comparison Table | `model_comparison.csv` |
| Presentation (10 slides) | `LSTM_Paper_Presentation.pptx` |
| Literature Review | `Literature_Review_LSTM_GRU.docx` |
| README | `README.md` |

---

## Research Paper Presentation

**Paper:** Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)

Covers: Problem Statement, Vanishing Gradient, LSTM Architecture & Gates, Experimental Results, Advantages/Limitations, Modern Applications, Personal Insights.

---

## Literature Review

**Paper:** LSTM-GRU Based Efficient Intrusion Detection in 6G-Enabled Metaverse Environments (2024)

Covers: Research Problem, Dataset, Methodology, Architecture, Results, Limitations, Future Work, Critical Analysis.

---

## Requirements

```
torch >= 2.0
pandas
numpy
scikit-learn
tqdm
```

---

**Deep Learning Intern**  
Week 4 – Sequential Modeling Assignment
