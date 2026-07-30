# Tesla Stock Price Prediction using RNN, LSTM, and GRU
### Sequence Modeling for Time-Series Forecasting

**Prepared by:** Your Name, Alishba & Mommna

---

## Overview

This project forecasts **Tesla (TSLA) closing stock price** using three recurrent neural network architectures — **Simple RNN**, **LSTM**, and **GRU** — implemented in PyTorch. The notebook (`stock_price_prediction_using_LSTM___RNN.ipynb`) walks through the full pipeline: data exploration, preprocessing, model development, training, evaluation, and performance comparison.

---

## 1. Topic: What is Time-Series Forecasting?

**Definition:** Predicting future values based on historical sequential data.

**Key Properties:**
- **Temporal Dependency** — each point depends on previous points, unlike i.i.d. tabular data.
- **Limited Future Data** — the model can never see the future during training; only the past is available.

This project treats Tesla's daily closing price as a univariate time series and uses a **60-day sliding window** of past prices to predict the next day's closing price (sequence-to-one forecasting).

---

## 2. Dataset

**Source:** `tesla_stock_price_14_years.csv`

| Property | Value |
|---|---|
| Rows | 3,432 daily records |
| Date Range | 2010-06-29 → 2024-02-15 |
| Columns | Date, Open, High, Low, Close, Adj Close, Volume |
| Target variable | `Close` price |
| Price range | $1.05 – $411.47 |

### Data Exploration & Visualization
Exploratory analysis (line plots, distribution, and correlation views) was performed before modeling to understand trend, volatility, and structure in the data:

![EDA Visualization](images/eda_visualization.png)

### Data Normalization
Neural networks train more stably when inputs are scaled. `MinMaxScaler` compresses the closing price into **[0, 1]**:

$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

![Data Normalization](images/data_normalization.png)

### Train-Test Split
- **Training set (80%)** — used to fit the models
- **Testing set (20%)** — held out, unseen data for evaluation
- **Chronological split, no shuffling** — order matters in time series; shuffling would leak future information into training

---

## 3. Model Development

All three models take the previous **60 days** of closing prices as input and predict the next day's closing price.

### Simple RNN
The most basic recurrent architecture. It maintains a hidden state that carries information from previous time steps:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

![RNN Architecture](images/rnn_architecture.png)

### LSTM (Long Short-Term Memory)
Adds a **memory cell** and three gates (forget, input, output) to overcome the vanishing-gradient problem and retain long-term dependencies:

$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f) \qquad i_t=\sigma(W_i[h_{t-1},x_t]+b_i) \qquad o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$$

![LSTM Architecture](images/lstm_architecture.png)

### GRU (Gated Recurrent Unit)
A simplified alternative to LSTM that merges the hidden state and memory cell into one, using only two gates (reset, update) — fewer parameters, often faster to train:

$$r_t=\sigma(W_r[h_{t-1},x_t]) \qquad z_t=\sigma(W_z[h_{t-1},x_t])$$

![GRU Architecture](images/gru_architecture.png)

**Shared hyperparameters:** hidden size = 64, num layers = 2, epochs = 50, learning rate = 0.001.

---

## 4. Model Evaluation

### Training & Validation Loss
![Training Loss Curves](images/training_loss_curves.png)

### Actual vs. Predicted Prices
![Actual vs Predicted](images/actual_vs_predicted.png)

---

## 5. Performance Comparison

| Model | MAE ($) | MSE | RMSE ($) | R² Score | Training Time (s) |
|---|---|---|---|---|---|
| **RNN** | 30.92 | 1,104.10 | 33.23 | 0.6725 | 427.14 |
| **LSTM** | 210.97 | 47,293.18 | 217.47 | **−13.03** | 761.94 |
| **GRU** | **15.03** | **469.15** | **21.66** | **0.8608** | 1,753.26 |

![Metrics Comparison](images/metrics_comparison.png)

### Discussion

- **Best accuracy: GRU** — lowest RMSE ($21.66) and highest R² (0.8608, explaining ~86% of price variance).
- **Fastest training: RNN** — simplest architecture, fewest computations per step, but weakest accuracy (R² = 0.67) since a plain RNN struggles to retain information across a 60-day window (vanishing gradients).
- **LSTM underperformed unexpectedly** in this run — its strongly negative R² (−13.03) indicates the model's predictions deviated far more from actual prices than simply predicting the mean would. This is not typical LSTM behavior and most likely points to a **training instability** in this run (e.g. exploding gradients, an unlucky weight initialization, or the learning rate being too high for LSTM's gate structure) rather than a fundamental weakness of LSTM itself. Recommended next step: retrain LSTM with gradient clipping, a lower learning rate, or a different random seed to confirm whether this was a one-off instability.
- **Practical takeaway:** for this dataset and configuration, GRU offered the best balance of accuracy; RNN offered speed but weaker accuracy; LSTM would need re-tuning before being trusted for production use.

---

## 6. Why Sequence Length (60 Days) Matters

| Window | Trade-off |
|---|---|
| Shorter (30 days) | Faster training, but less pattern captured → lower accuracy |
| **60 days (used here)** | ~3 months of history — captures quarterly trends and short seasonality, good balance |
| Longer (180 days) | More context, but harder to train and higher overfitting risk |

---

## 7. Presentation — "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (Chung et al., 2014)

A 9-slide, diagram-first presentation on the GRU paper is included:
**`GRU_Chung2014_Presentation.pptx`**

Slides: Title → Motivation → GRU Architecture → GRU vs. LSTM → Experimental Setup → Experimental Results → Advantages & Limitations → Practical Applications → Conclusion.

## 8. Literature Review — Recent Paper (2023–2026)

A 2-page critical review is included: **`Paper_Review_ANTM_Robustness.docx`**

**Paper reviewed:** Khoirudin, Pungkasanti, P. T., Wakhidah, N., & Rishiwal, V. (2026). *A Robustness-Oriented Evaluation of LSTM, GRU, and Hybrid LSTM-GRU Models for ANTM.JK Stock Price Forecasting.* Journal of Information Systems and Informatics, 8(3), 3734–3757. https://doi.org/10.63158/journalisi.v8i3.1660

Covers: Research Problem, Dataset, Model Architecture, Results, Strengths, Limitations, Future Work, and Critical Analysis.

---

## Deliverables Checklist
- ✅ Jupyter Notebook — `stock_price_prediction_using_LSTM___RNN.ipynb`
- ✅ Saved Models (`.pth`) — RNN, LSTM, GRU state dicts saved at the end of the notebook
- ✅ Diagrams — `images/` (EDA, normalization, architectures, loss curves, predictions, metrics comparison)
- ✅ Presentation — Chung et al. 2014 GRU paper (9 slides) — `GRU_Chung2014_Presentation.pptx`
- ✅ Literature Review (2026, 2 pages) — `Paper_Review_ANTM_Robustness.docx`
- ✅ README — this file

## How to Run
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
jupyter notebook stock_price_prediction_using_LSTM___RNN.ipynb
```
