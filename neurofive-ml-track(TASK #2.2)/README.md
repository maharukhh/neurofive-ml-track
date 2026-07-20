# Week 2, Task 2 — House Price Prediction with Linear Regression

**Neurofive ML Track**

## Overview
This task introduces **regression** — predicting a continuous number
(median house value) rather than a category. We use the California Housing
dataset (1990 census, block-group level), the standard modern replacement
for the deprecated Boston Housing dataset.

## Notebook
- `house_price_regression.ipynb`

## Approach

1. **Missing values:** only `total_bedrooms` had gaps (207 of 20,640 rows,
   ~1%) — filled with the median rather than dropping rows, since the loss
   would outweigh the benefit for such a small fraction.

2. **Feature selection (5 features):**
   - `median_income` — area income strongly relates to affordability/price
   - `total_rooms` — proxy for the scale of housing in the area
   - `housing_median_age` — older vs. newer housing stock
   - `latitude` / `longitude` — location, one of the biggest drivers of California real estate prices

3. **Train/test split:** 80/20 with `train_test_split(random_state=42)`.

4. **Model:** `LinearRegression` from scikit-learn.

5. **Evaluation:** RMSE (Root Mean Squared Error) and R² score, plus a
   predicted-vs-actual scatter plot and a residual plot.

## Results

- **RMSE: ~$73,793** — on average, predictions are off by about this much in dollar terms.
- **R² score: 0.5844** (~58.4%)

**Predicted vs. Actual plot:** points cluster around the "perfect
prediction" line for mid-range prices but spread out more at the high end —
the model tends to underestimate the most expensive homes. There's also a
visible flat ceiling near $500,000, which comes from the dataset itself
(house values were capped at that amount during data collection).

**Residual plot:** shows a slight fan/curve pattern rather than a random
scatter around zero, suggesting a plain linear model doesn't fully capture
the relationship — a more flexible model (polynomial features or a
tree-based model) could likely do better.

## What does the R² score mean? (Plain English)

Our model's R² score of 0.58 means that roughly **58% of the differences in
house prices across neighborhoods can be explained by the five factors we
gave the model** — income level, number of rooms, house age, and location.
The remaining 42% comes from things the model doesn't know about, like the
condition of a specific home, school quality, or crime rates. In short: the
model gives a reasonable ballpark estimate of what a house in an area
*should* be worth, but it isn't precise enough to price an individual home
exactly — think of it as a rough neighborhood-level estimate, not a formal
appraisal.

## Tools Used
- Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook
