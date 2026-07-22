# Week 4, Task 2 — Ensemble Learning: Random Forest vs. XGBoost

**Neurofive ML Track**

## Overview
Compares two ensemble methods -- **Random Forest** (bagging) and
**XGBoost** (boosting) -- against the earlier single Logistic Regression
model on the Titanic dataset, reusing the engineered features
(`FamilySize`, `IsAlone`, `Title`) from the pipeline task.

## Notebook
- `titanic_ensemble_models.ipynb`

## Approach

1. Reused the same cleaning and feature engineering as the Week 4 Pipeline task.
2. Trained three models on an 80/20 train/test split:
   - **Logistic Regression** (single model, baseline)
   - **Random Forest** (`n_estimators=200`, `max_depth=6`)
   - **XGBoost** (`n_estimators=200`, `max_depth=4`, `learning_rate=0.1`)
3. Compared all three in one table (model / metric / score).
4. Plotted and compared `.feature_importances_` for Random Forest and XGBoost side by side.

## Results

| Model                              | Accuracy | Precision (Survived) | Recall (Survived) | F1-score (Survived) |
|-------------------------------------|:--------:|:---------------------:|:-------------------:|:---------------------:|
| Logistic Regression (single model) | 0.8212   | 0.79                  | 0.72                | 0.75                  |
| **Random Forest (ensemble)**        | **0.8268** | 0.82                | 0.71                | 0.76                  |
| XGBoost (ensemble)                  | 0.7989   | 0.75                  | 0.71                | 0.73                  |

**Honest takeaway:** Random Forest gave the best accuracy here, a modest
improvement over the single Logistic Regression model. XGBoost actually
underperformed both on this run, which is a real and common outcome on a
small dataset like Titanic (~900 rows) — boosting models have more
hyperparameters and more capacity to overfit the training data without
careful tuning (learning rate, tree depth, regularization, early stopping),
so out-of-the-box settings don't always beat simpler models on small data.
This is a useful lesson in itself: ensemble methods are powerful, but not
automatically better without proper tuning.

## Feature Importances: Random Forest vs. XGBoost

**Random Forest top 5:** `Title_Mr`, `Sex_male`, `Fare`, `Pclass`, `Age`
**XGBoost top 5:** `Title_Mr`, `Pclass`, `Title_Rare`, `HasCabin`, `FamilySize`

Both models agree `Title_Mr` is the single most important feature (carrying
most of the "women and children first" signal). Beyond that, Random Forest
spreads importance more evenly across several features (`Sex_male`, `Fare`,
`Age` all contribute meaningfully), while XGBoost concentrates much more
heavily on `Title_Mr` alone (0.43 vs. Random Forest's 0.21) and leans on a
different secondary set of features.

## How Random Forest and XGBoost differ in combining models

**Random Forest** builds many decision trees **independently and in
parallel** — each tree trains on a random bootstrap sample of rows and
considers only a random subset of features at each split. The final
prediction is a majority vote (classification) or average (regression)
across all trees — this is **bagging**, and it reduces variance by
averaging out individual trees' mistakes. **XGBoost** builds trees
**sequentially**, where each new tree is trained specifically to correct the
errors of the trees built so far (**boosting**) — rather than voting
independently, each tree adds a small weighted correction on top of the
running prediction. Boosting can reach higher accuracy than bagging when
well-tuned, but it's more sensitive to overfitting and must be trained in
strict sequence rather than in parallel, which is exactly what we observed
here with the untuned XGBoost model underperforming.

## Tools Used
- Python, Pandas, NumPy, scikit-learn (`LogisticRegression`, `RandomForestClassifier`), XGBoost (`XGBClassifier`), Matplotlib, Seaborn, Jupyter Notebook
