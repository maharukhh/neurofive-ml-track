# Week 3, Task 1 — Model Evaluation and Hyperparameter Tuning

**Neurofive ML Track**

## Overview
This task revisits the Titanic Logistic Regression classifier from Week 2 to
evaluate it properly with precision/recall/F1 (not just accuracy), explain
why accuracy alone is misleading on imbalanced data, and tune it
systematically with `GridSearchCV`.

## Notebook
- `titanic_model_tuning.ipynb`

## Approach

1. **Re-used the same cleaning and features** as the Week 2 classification task (grouped-median `Age`, mode `Embarked`, `HasCabin` flag, one-hot encoded `Sex`/`Embarked`).
2. **Checked class balance:** `Survived` is ~61.6% "No" / ~38.4% "Yes" — a moderate imbalance.
3. **Computed precision, recall, and F1-score** with `classification_report`.
4. **Demonstrated why accuracy is misleading:** built a dummy "always predict did not survive" baseline that scores **61.45% accuracy** while having **0.00 recall and 0.00 F1-score** for the "Survived" class — proof that a high accuracy number can hide a model that never correctly identifies the minority class.
5. **Tuned hyperparameters with `GridSearchCV`:**
   - `C` (regularization strength): `[0.01, 0.1, 1, 10, 100]`
   - `penalty` (regularization type): `['l1', 'l2']`
   - 5-fold cross-validation, optimizing for **F1-score** (not accuracy, deliberately — see point 4).
6. **Compared baseline vs. tuned model** in a before/after table and side-by-side confusion matrices.

## Why accuracy alone can be misleading (in my own words)

Accuracy treats every correct/incorrect prediction the same, regardless of
which class it came from. On an imbalanced dataset, a model can rack up a
deceptively high accuracy just by leaning on the majority class and barely
engaging with the minority class. Here, a "model" that predicts "did not
survive" for every single passenger — without learning anything — still
scores 61.45% accuracy, yet it never correctly identifies a single survivor
(0.00 recall, 0.00 F1 for "Survived"). That's exactly why precision, recall,
and F1-score matter more than accuracy alone: they expose per-class
performance, so a model that ignores the minority class gets caught rather
than hidden behind one misleadingly reasonable-looking number.

## Results — Before vs. After Tuning

| Metric                | Baseline Model | Tuned Model | Change |
|------------------------|:--------------:|:-----------:|:------:|
| Accuracy               | 0.8101          | 0.8101       | 0.0000 |
| Precision (Survived)    | 0.78            | 0.78         | 0.0000 |
| Recall (Survived)       | 0.71            | 0.71         | 0.0000 |
| F1-score (Survived)     | 0.74            | 0.74         | 0.0000 |

**Best hyperparameters found:** `C=100`, `penalty='l1'`, `solver='liblinear'`
(best cross-validated F1-score: 0.7401)

**Honest takeaway:** in this case, `GridSearchCV` landed on hyperparameters
that perform identically to the untuned baseline on the test set. This is a
real and fairly common outcome — for a simple, already well-behaved model
like Logistic Regression on a small, mostly-linear-separable dataset like
Titanic, the default settings were already close to optimal, so tuning
mainly confirms the model is well-calibrated rather than producing a big
jump in performance. Tuning tends to matter more for models with more
hyperparameters and more room to overfit (e.g., tree-based models, SVMs with
different kernels), which is worth exploring in a follow-up task.

## Tools Used
- Python, Pandas, NumPy, scikit-learn (`GridSearchCV`, `classification_report`), Matplotlib, Seaborn, Jupyter Notebook
