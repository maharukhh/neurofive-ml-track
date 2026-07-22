# Week 3 — Customer Churn Prediction: Working with a Business Problem

**Neurofive ML Track**

## Overview
Predicts whether a telecom customer will churn (cancel service), using the
IBM Telco Customer Churn dataset (7,043 customers). This task introduces
**Decision Trees**, compares them against Logistic Regression, and closes
with a plain-English business summary.

## Notebook
- `telco_churn_prediction.ipynb`

## Approach

1. **Cleaning:**
   - `TotalCharges` was loaded as text due to 11 blank values (all belonging to brand-new customers with `tenure = 0`) — converted to numeric and filled with 0.
   - `customerID` dropped (identifier, not a feature).

2. **EDA — what correlates with churn:**
   - **Contract type** shows the clearest gap: month-to-month customers churn far more than one-/two-year contract customers.
   - **Tenure** — churned customers tend to be newer customers; long-tenured customers rarely leave.
   - **Monthly charges** — churned customers tend to pay somewhat more per month.

3. **Class imbalance:** ~73.5% "No churn" vs. ~26.5% "Churn". This is flagged
   but not specifically corrected (no SMOTE/class-weighting applied) — instead,
   precision/recall/F1 are used to judge the models fairly rather than relying
   on accuracy alone, since a model could reach ~73% accuracy by just
   predicting "No churn" for everyone.

4. **Encoding:** binary Yes/No columns (and `gender`) mapped to 0/1; multi-category columns (`Contract`, `PaymentMethod`, `InternetService`, etc.) one-hot encoded with `pd.get_dummies()`.

5. **Models trained:** Logistic Regression and a depth-limited (`max_depth=5`) Decision Tree, to keep the tree interpretable rather than overfit.

## Results

| Model                | Accuracy | Precision (Churn) | Recall (Churn) | F1-score (Churn) |
|-----------------------|:--------:|:------------------:|:---------------:|:------------------:|
| Logistic Regression   | 0.8048   | 0.66                | 0.56             | 0.60               |
| Decision Tree         | 0.7942   | 0.63                | 0.54             | 0.58               |

Logistic Regression edged out the Decision Tree slightly on every metric
here. The Decision Tree's real advantage isn't raw performance — it's
**interpretability**: we can trace the exact yes/no splits that lead to a
churn prediction and read off feature importances directly, both far easier
to explain to a non-technical audience than logistic regression
coefficients.

## Top 3 Features Driving Churn (Decision Tree `.feature_importances_`)

1. **`tenure`** (0.42) — how long the customer has been with the company
2. **`InternetService_Fiber optic`** (0.36) — whether the customer has fiber internet
3. **`TotalCharges`** (0.04) — total amount billed to date

(Note: `Contract` type showed a strong relationship with churn in the raw
EDA crosstab, but its importance is split across several one-hot encoded
columns in the tree, so no single `Contract_*` column ranks in the top 3
individually — the business summary below accounts for this.)

## Business Summary

We built a model to flag customers who are likely to cancel their service,
and the single biggest driver is how long someone has been a customer
(**tenure**) — brand-new customers are far more likely to churn than
long-standing ones, so the first several months of the relationship are the
highest-risk window and deserve extra retention attention (onboarding
check-ins, early loyalty perks). The second strongest signal is having
**fiber optic internet service** — these customers churn noticeably more
than DSL or no-internet customers, which may point to pricing or service
reliability issues worth investigating with that product team. Our own
exploration also showed that customers on flexible **month-to-month
contracts** churn far more than those on one- or two-year contracts, so
incentivizing longer commitments (even with a modest discount) is a
practical lever the tree's ranking doesn't fully capture on its own, since
that single business factor gets split across several technical columns
behind the scenes. Overall, the model correctly identifies roughly 8 in 10
customers' churn status, giving the retention team a practical early-warning
list rather than a perfect crystal ball — pairing it with human judgment on
high-value accounts would likely deliver the most value.

## Tools Used
- Python, Pandas, NumPy, scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`), Matplotlib, Seaborn, Jupyter Notebook
