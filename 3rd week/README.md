# Artificial Neural Networks — From Scratch to TensorFlow

A hands-on implementation of Artificial Neural Networks (ANNs), built once from scratch in NumPy and once in TensorFlow/Keras, with outputs cross-validated against each other. The project closes with an applied classifier trained on a real medical dataset.

---

## Overview

This repository documents a complete ANN workflow, from a single neuron to a trained classifier:

- A single artificial neuron implemented manually (weighted sum + activation)
- A full `2 → 3 → 1` feedforward network, with forward propagation computed step by step in NumPy
- The identical network rebuilt in TensorFlow/Keras, loaded with the same weights, to confirm both implementations produce matching output
- Six activation functions (Binary Step, Sigmoid, Tanh, ReLU, Leaky ReLU, ELU) implemented from scratch and plotted
- An applied classifier trained on the Breast Cancer Wisconsin dataset, evaluated with accuracy, loss curves, and a confusion matrix

A companion slide deck, `ANN_Presentation.pptx`, summarizes the theory and results for presentation purposes.

---

## Repository Structure

```
.
├── ANN_Notebook.ipynb        Main implementation notebook
├── ANN_Presentation.pptx     Slide deck summarizing theory and results
└── README.md
```

---

## Notebook Contents — ANN_Notebook.ipynb

| Task | Description |
|------|-------------|
| 1 | Import core libraries — NumPy, Matplotlib, TensorFlow/Keras, scikit-learn |
| 2 | Implement activation functions from scratch (Binary Step, Sigmoid, Tanh, ReLU, Leaky ReLU, ELU) and plot each |
| 3 | Build a single artificial neuron: weighted sum followed by an activation |
| 4 | Implement full forward propagation for a `2 → 3 → 1` network in raw NumPy |
| 5 | Rebuild the same architecture in TensorFlow/Keras, load identical weights, and verify the outputs match |
| 6 | Plot an overlayed comparison of all activation functions |
| 7 (Bonus) | Train an end-to-end ANN classifier on the Breast Cancer Wisconsin dataset |

### Bonus Task — Breast Cancer Classifier

**Architecture:** `30 → 16 → 8 → 1`, ReLU hidden layers, Sigmoid output layer
**Optimizer / Loss:** Adam, binary cross-entropy

| Metric | Value |
|---|---|
| Test Accuracy | 94.74% |
| Test Loss | 0.1207 |

Training and validation loss curves track closely across epochs, indicating the model generalizes well without significant overfitting. The full training curves and confusion matrix are available in the notebook output.

---

## Presentation — ANN_Presentation.pptx

A 21-slide deck that walks through the theory and results in presentation form:

| # | Slide | Highlights |
|---|-------|-----------|
| 1 | Title | Artificial Neural Networks — Foundations, Architecture, Mathematics & Applications |
| 2 | Agenda | Overview of the 9 topics covered in the deck |
| 3 | Introduction | AI → Machine Learning → Deep Learning, and what makes an ANN different from rule-based systems |
| 4 | History | Timeline from the 1943 McCulloch–Pitts neuron to today's deep architectures |
| 5–6 | Biological Inspiration | Dendrites, soma, axon, synapse — mapped to inputs, weights, summation, and activation |
| 7–8 | The Perceptron | 1958 Rosenblatt model, its decision rule, and its strengths/limitations (fails on XOR) |
| 9 | Architecture | Input, hidden, and output layers in a fully-connected network |
| 10 | Mathematical Foundation | Weighted sum and activation, in both scalar and matrix form |
| 11–12 | Activation Functions | Why non-linearity is required, with a visual comparison of functions |
| 13–14 | Forward Propagation | How data flows layer by layer, plus a fully worked numerical example (inputs 0.6 / 0.9 → final prediction ≈ 0.5135) |
| 15 | Hyperparameters | Learning rate, batch size, epochs, hidden layers, neurons per layer |
| 16 | Advantages & Limitations | Trade-offs of ANNs as a modeling approach |
| 17 | Applications | Image classification, healthcare, finance, speech, recommendations, fraud detection, forecasting |
| 18 | Training Pipeline | The bigger picture — how the pieces fit together end to end |
| 19 | Notebook Walkthrough | Slide-by-slide summary of Tasks 1–7 in `ANN_Notebook.ipynb` |
| 20 | Results Snapshot | Bonus classifier results — `30 → 16 → 8 → 1` Keras model, converges within ~50 epochs |
| 21 | Closing | Thank you / wrap-up |

---

## Core Math

**Single neuron:**

```
z = Σ(wᵢxᵢ) + b
a = f(z)
```

**Layer, in matrix form:**

```
Z = W·X + b
A = f(Z)
```

**Perceptron learning rule:**

```
wᵢ ← wᵢ + η(y_true − y_pred)xᵢ
```

**Forward propagation, 2 → 3 → 1 network:**

```
Z1 = W1·X + b1,  A1 = tanh(Z1)
Z2 = W2·A1 + b2, A2 = sigmoid(Z2)
```

---

## Requirements

- Python 3.10 or later
- numpy
- matplotlib
- tensorflow
- scikit-learn

Open `ANN_Notebook.ipynb` in Jupyter, JupyterLab, or Google Colab to run it.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Mommna Ahmed
