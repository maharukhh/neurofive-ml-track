# Week 2, Task 1 — Predict Titanic Survival: First Classification Model

**Neurofive ML Track**

## Overview
This task builds the first machine learning model of the track: a **Logistic
Regression classifier** that predicts whether a Titanic passenger survived
(`1`) or did not (`0`), using the cleaned dataset from the earlier EDA task.

## Notebook
- `titanic_classification.ipynb`

## Approach

1. **Cleaning (carried over from EDA):**
   - `Age` → filled with the median, grouped by `Pclass` and `Sex`
   - `Embarked` → filled with the mode
   - `Cabin` → converted to a `HasCabin` flag; the raw column was dropped (too sparse to use directly)

2. **Feature selection:** dropped `PassengerId`, `Name`, and `Ticket` since
   they are identifiers/free text not directly usable by a simple model.

3. **Encoding:** `Sex` and `Embarked` were one-hot encoded with
   `pd.get_dummies(..., drop_first=True)`, since scikit-learn models require
   numeric input.

4. **Train/test split:** used `train_test_split` with `test_size=0.2`
   (80/20 split), `random_state=42` for reproducibility, and
   `stratify=target` so both sets keep the same survival ratio as the full
   dataset (~38% survived).

5. **Model:** `LogisticRegression` from scikit-learn (`max_iter=1000` to
   ensure convergence).

6. **Evaluation:** `accuracy_score`, a confusion matrix, and a full
   classification report (precision/recall/F1).

## Results

- **Accuracy: [apna accuracy number yahan daalo]%**

**Confusion Matrix:**

|                        | Predicted: Did Not Survive | Predicted: Survived |
|------------------------|:---------------------------:|:--------------------:|
| **Actual: Did Not Survive** | [TN]      | [FP]  |
| **Actual: Survived**        | [FN]     | [TP]   |

**What this tells us:**
- The confusion matrix breaks predictions into four outcomes: correctly
  predicted non-survivors, false alarms (predicted survived but didn't),
  missed survivors (predicted did not survive but actually did), and
  correctly predicted survivors.
- [Apne actual numbers dekh kar likho: False Negatives zyada hain ya False
  Positives — us hisaab se model kis taraf bias hai.]

**Most influential features:** consistent with the earlier EDA, `Sex`
(specifically being male) had the strongest negative effect on survival
probability, followed by `Pclass` (higher class number → lower survival
chance).

## Tools Used
- Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook
