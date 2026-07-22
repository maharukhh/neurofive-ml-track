# Week 4, Task 1 — Build a Proper ML Pipeline with Feature Engineering

**Neurofive ML Track**

## Overview
Moves from manual, cell-by-cell preprocessing to a single, reusable
**scikit-learn Pipeline** using `ColumnTransformer`, and adds engineered
features to the Titanic dataset to test whether they improve performance.

## Notebook
- `titanic_pipeline.ipynb`
- `titanic_pipeline.joblib` — the final saved, fitted pipeline

## Approach

1. **Manual baseline:** re-ran the earlier manual preprocessing approach (hand-written `fillna`, `pd.get_dummies`) as a point of comparison.

2. **Feature engineering (3 new features):**
   - `FamilySize` = `SibSp` + `Parch` + 1
   - `IsAlone` = 1 if `FamilySize == 1`, else 0
   - `Title` = extracted from the `Name` column (Mr / Mrs / Miss / Master / Rare)

3. **Pipeline construction with `ColumnTransformer`:**
   - **Numerical columns** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `IsAlone`, `HasCabin`) → `SimpleImputer(median)` → `StandardScaler`
   - **Categorical columns** (`Sex`, `Embarked`, `Title`) → `SimpleImputer(most_frequent)` → `OneHotEncoder`
   - Combined via `ColumnTransformer`, then chained with `LogisticRegression` into one `Pipeline` object.

4. **Why this avoids data leakage:** because the imputer/scaler/encoder live *inside* the pipeline, calling `.fit()` on the training set only ever computes statistics (medians, means, categories) from that training set. The test set only ever gets *transformed* using those already-fitted values, never used to refit anything — unlike a manual approach, where it's easy to accidentally scale or impute using the full dataset before splitting.

5. **Saved the final pipeline** with `joblib.dump()`, then reloaded it and confirmed it reproduces identical predictions.

## Results

| Approach                                                         | Accuracy |
|-------------------------------------------------------------------|:--------:|
| Manual preprocessing (hand-written `fillna`, `pd.get_dummies`)     | 0.8101   |
| Pipeline, same features, no feature engineering                   | 0.8101   |
| **Pipeline, WITH engineered features** (`FamilySize`, `IsAlone`, `Title`) | **0.8324** |

**Takeaways:**
- The pipeline exactly matches the manual approach's accuracy when using the
  same features — confirming it correctly replicates the same logic while
  being structurally safer against data leakage.
- Adding the three engineered features improved accuracy from 0.8101 to
  **0.8324** (about +2.2 percentage points), and improved precision/recall
  for the "Survived" class too (F1-score for "Survived" rose from 0.74 to
  0.78). `Title` in particular adds real signal beyond `Sex`/`Age` alone —
  e.g., "Master" (young boys) survived at 57.5%, much higher than adult "Mr"
  passengers at 15.7%, a distinction `Sex` alone can't capture.

## Tools Used
- Python, Pandas, NumPy, scikit-learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `LogisticRegression`), joblib, Jupyter Notebook
