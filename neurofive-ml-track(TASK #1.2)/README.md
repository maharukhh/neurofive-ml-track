# Task 1.2 — Data Cleaning & Visualization (Titanic Dataset)

**Neurofive ML Track**

## Overview
This task focuses on cleaning the Titanic dataset and creating visualizations
to understand its patterns before any model training.

## Notebook
- `titanic_eda_cleaning.ipynb`

## What Was Done

1. **Handled Missing Values:**
   - `Age` → filled using the median, grouped by `Pclass` and `Sex`
   - `Embarked` → filled using the mode (most frequent value)
   - `Cabin` → 77% missing, so converted into a `HasCabin` flag and dropped the original column

2. **Outlier Detection:**
   - Used boxplots on `Fare` and `Age` to visually spot outliers
   - Applied the IQR method to count Fare outliers

3. **Visualizations Created:**
   - Histogram — Age distribution
   - Boxplot — Fare by Passenger Class
   - Bar Chart — Survival Rate by Sex
   - Correlation Heatmap — relationships between numeric features

## Findings
- **Sex** is the strongest predictor of survival (females ~74% survival
  rate, males ~19% survival rate).
- **Pclass** is the second most important factor — 1st class passengers
  had a higher survival rate than 2nd or 3rd class.
- Outliers in `Fare` are legitimate high-value 1st class tickets, not
  data errors.

## Tools Used
- Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook (VS Code)