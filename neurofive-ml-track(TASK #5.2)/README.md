# Week 5, Task 2 — Deploy Your Model as a Live Web App

**Neurofive ML Track**

## Overview
Takes the best-performing model so far -- the Titanic survival prediction
`Pipeline` from the Week 4 Pipeline task (Logistic Regression +
`ColumnTransformer` with engineered features, **~83% test accuracy**) -- and
wraps it in a simple, shareable **Streamlit** web app.

## Files
- `app.py` -- the Streamlit app
- `titanic_pipeline.joblib` -- the saved, fitted pipeline (from Week 4, Task 1)
- `requirements.txt` -- dependencies needed to run/deploy the app

## What the app does

1. Loads the saved pipeline with `joblib.load()` (cached with `@st.cache_resource` so it only loads once per session, not on every interaction).
2. Shows input fields for the key passenger details: class, sex, age, title, siblings/spouses aboard, parents/children aboard, fare, port of embarkation, and whether a cabin was recorded.
3. Automatically computes the two engineered features (`FamilySize`, `IsAlone`) from the raw inputs, exactly the way they were computed during training -- the user never has to enter them manually.
4. On clicking **Predict**, builds a one-row DataFrame matching the exact column names/order the pipeline expects, calls `.predict()` and `.predict_proba()`, and displays the prediction with a confidence percentage and progress bar.

**Sanity-checked predictions before deployment:**
- A young 1st-class woman → predicted **survived**, 94.5% confidence.
- An adult 3rd-class man → predicted **did not survive**, 93.5% confidence (6.5% survival probability).

Both match the well-known "women and children first" / class-based survival
pattern from the EDA in earlier tasks.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser, usually at `http://localhost:8501`.

## Deploying for free on Streamlit Community Cloud

 **the link to your main repo README** 
   
   ## 🔗 Live App
   https://neurofive-ml-track.streamlit.app/

## Tools Used
- Python, Streamlit, scikit-learn, Pandas, joblib
