# Regularization, Hyperparameter Optimization & Training Strategies for ANNs

A from-scratch (NumPy-only) implementation of a feed-forward neural network, used to study how
different regularization and optimization techniques affect training stability and
generalization.

## Objective

Practice and compare the core techniques used to improve the performance, stability, and
generalization of Artificial Neural Networks:

- Dropout
- Batch Normalization
- Weight initialization (Random / Xavier / He)
- Early stopping
- Model checkpointing (restore best weights)
- Learning-rate scheduling
- Systematic hyperparameter search (learning rate, depth, width, batch size)

The neural network engine (forward pass, backpropagation, parameter updates) is implemented
manually with NumPy — no TensorFlow/Keras/PyTorch is used for the model itself, so every layer's
math is fully visible and inspectable.

## Dataset

**Optical Recognition of Handwritten Digits** (`sklearn.datasets.load_digits`)

- 1,797 samples, 64 numeric features (8x8 grayscale pixel intensities, values 0–16)
- Ships built into scikit-learn — loads instantly, no download required
- Reframed as a **binary classification** problem: *"Is this handwritten digit an 8, or not?"*
  (~10% of samples are positive, so the task is moderately imbalanced)
- Split: 60% train / 20% validation / 20% test, stratified on the label
- Features standardized using training-set statistics only (`StandardScaler`)

## Requirements

```
python >= 3.9
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run

```bash
jupyter notebook ANN_Regularization_HyperparamOpt_Digits.ipynb
```

Then run all cells top to bottom (`Kernel → Restart & Run All`). Each task cell can also be run
independently once the dataset-loading and engine cells have been executed.

## Notebook Structure

| Section | What it covers |
|---|---|
| Setup | Load digits dataset, build binary target, train/val/test split, standardization |
| NumPy engine | `DenseLayer`, `ReLU`, `Sigmoid`, `DropoutLayer`, `BatchNormLayer`, `NeuralNet` wrapper |
| Training loop | Mini-batch SGD with optional LR schedule, early stopping, and checkpointing |
| Task 1 | Baseline ANN (`64 → 32 → 16 → 1`, He init, no dropout/batchnorm) |
| Task 2 | Dropout comparison: none vs. 0.3 vs. 0.5 |
| Task 3 | Batch Normalization: off vs. on |
| Task 4 | Weight initialization: Random vs. Xavier vs. He |
| Task 5 | Early stopping + checkpointing + step-decay LR schedule vs. plain fixed-LR training |
| Task 6 | Hyperparameter sweeps: learning rate, network depth, layer width, batch size |
| Task 7 | Consolidated visualizations across all sweeps |
| Conclusions | Summary takeaways from every task |

## Key Takeaways

- The train/validation **loss gap** is a more reliable overfitting signal than raw accuracy here,
  since the "is this an 8?" task is imbalanced.
- Dropout reduces the train/val gap but can underfit a small network if set too high (0.5).
- Batch Normalization mainly speeds up/stabilizes convergence rather than boosting final accuracy
  on a dataset this size.
- He initialization is the most reliable choice for this ReLU-based network.
- Early stopping + checkpointing recovers the best-validation-loss weights automatically, removing
  the need to guess a fixed epoch count.
- Every hyperparameter (learning rate, depth, width, batch size) shows a clear sweet spot — too
  small/shallow underfits, too large/deep adds cost without a matching validation benefit.

## Notes

- All results in the notebook were generated with `RANDOM_STATE = 7` for reproducibility.
- This notebook uses a different dataset (Digits, 64 features, image-based) than the original
  reference notebook (Breast Cancer, 30 features, tabular) while following the same task
  structure, so the two can be compared side by side without being identical.
