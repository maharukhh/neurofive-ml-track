# Week 5, Task 1 — Handling Imbalanced & Messy Real-World Data

**Neurofive ML Track**

## Overview
Uses the **Credit Card Fraud Detection** dataset (284,807 anonymized
real transactions, only 492 of which are fraud — **0.17%**) to demonstrate
severe class imbalance, why accuracy is a dangerous metric here, and how to
address it with SMOTE and class weighting.

## Notebook
- `credit_card_fraud_imbalance.ipynb`

## Dataset
- 284,807 transactions, 31 columns (`Time`, `Amount`, `V1`–`V28` PCA-anonymized features, `Class` target).
- No missing values.
- **Class balance: 99.83% normal / 0.17% fraud** — far more extreme than the Titanic (~38% survived) or churn (~26.5% churned) datasets used in earlier tasks.

## Why accuracy would be a badly misleading metric here

With fraud at just 0.17% of the data, a "model" that does zero learning and
predicts "not fraud" for every single transaction scores **99.83% accuracy**
— a number that looks almost perfect while being completely useless at
catching fraud. This was proven concretely in the notebook: that naive
baseline gets 99.83% accuracy but **0.00 recall and 0.00 F1-score** for the
fraud class. In a real business setting, this kind of misleadingly high
accuracy could give a fraud team false confidence that their system is
working, while every fraudulent transaction slips through undetected. This
is exactly why precision, recall, and F1-score — specifically for the fraud
class — are the metrics that actually matter here, not overall accuracy.

## Approach

1. Scaled `Time` and `Amount` with `StandardScaler` (the `V1`–`V28` columns are already PCA-scaled by the dataset's original authors).
2. Split into an 80/20 train/test set, **stratified** on `Class` to preserve the fraud ratio in both sets.
3. Trained a baseline Logistic Regression on the raw, imbalanced training data.
4. Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the *training set only* (never the test set — that would leak information and give a falsely optimistic result), generating synthetic fraud examples until the training classes were balanced 50/50.
5. Retrained on the SMOTE-balanced data and compared to baseline.
6. Also tried **`class_weight='balanced'`** as a second, lighter-weight technique for comparison (no synthetic data, just reweights the loss function).

## Results — Before vs. After

| Approach                          | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
|-------------------------------------|:--------:|:-------------------:|:-----------------:|:--------------------:|
| Baseline (raw imbalanced data)      | 0.9992   | 0.83                 | 0.64               | 0.72                 |
| **SMOTE oversampling**              | 0.9743   | 0.06                 | **0.92**           | 0.11                 |
| `class_weight='balanced'`           | 0.9755   | 0.06                 | **0.92**           | 0.11                 |

**Honest reading of these results:** both imbalance-handling techniques
dramatically increased **recall** for fraud, from 0.64 to 0.92 — catching far
more actual fraud cases. But this came at a steep cost to **precision**,
which dropped from 0.83 to 0.06 — meaning the balanced models flag many more
legitimate transactions as fraud (false alarms). Accuracy also dropped
overall (0.9992 → ~0.975), but that drop is *expected and acceptable* here —
it's a direct consequence of trading some false-alarm noise for much better
fraud coverage.

In a real fraud detection system, missing actual fraud (false negatives) is
usually far more costly than a false alarm that gets reviewed and dismissed
(false positives), so this tradeoff often makes business sense. That said, a
93% false-alarm rate on flagged transactions would overwhelm a real review
team — this notebook demonstrates the *technique* and its tradeoff clearly,
but a production system would need further tuning (e.g., adjusting the
classification threshold, combining SMOTE with undersampling, or using a
more sophisticated model) to land on a more balanced precision/recall point
than shown here.

## Tools Used
- Python, Pandas, NumPy, scikit-learn, imbalanced-learn (`SMOTE`), Matplotlib, Seaborn, Jupyter Notebook
