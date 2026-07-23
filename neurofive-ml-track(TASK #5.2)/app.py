"""
Titanic Survival Predictor -- Streamlit Web App
Neurofive ML Track -- Week 5, Task 2

Loads the saved scikit-learn pipeline (trained in Week 4, Task 1) and lets
a user enter passenger details through a simple UI to get a live survival
prediction.
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter a passenger's details below and the model will predict whether "
    "they would have survived the Titanic disaster. This app uses a "
    "Logistic Regression model trained inside a scikit-learn `Pipeline` "
    "(with `StandardScaler` + `OneHotEncoder` via `ColumnTransformer`) on "
    "the classic Titanic dataset, reaching about **83% accuracy** on held-out "
    "test data."
)

# ---------------------------------------------------------------------------
# Load the saved pipeline (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("titanic_pipeline.joblib")

pipeline = load_pipeline()

st.divider()

# ---------------------------------------------------------------------------
# Input fields for the key features
# ---------------------------------------------------------------------------
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x],
        index=2,
    )
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=80, value=28)
    title = st.selectbox(
        "Title",
        options=["Mr", "Mrs", "Miss", "Master", "Rare"],
        help="Mr/Mrs/Miss/Master come straight from the passenger's name; "
             "use 'Rare' for uncommon titles (e.g. Dr, Rev, Col).",
    )

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
    )

has_cabin = st.checkbox("Cabin number was recorded for this passenger", value=False)

# ---------------------------------------------------------------------------
# Derived / engineered features (computed automatically, same as training)
# ---------------------------------------------------------------------------
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

st.caption(f"Derived automatically: Family Size = {family_size}, Traveling Alone = {'Yes' if is_alone else 'No'}")

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "HasCabin": int(has_cabin),
        "Sex": sex,
        "Embarked": embarked,
        "Title": title,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    survival_prob = probability[1] * 100
    death_prob = probability[0] * 100

    if prediction == 1:
        st.success(f"### ✅ Predicted: SURVIVED")
        st.write(f"Model confidence: **{survival_prob:.1f}%** chance of survival")
    else:
        st.error(f"### ❌ Predicted: DID NOT SURVIVE")
        st.write(f"Model confidence: **{death_prob:.1f}%** chance of not surviving")

    st.progress(int(survival_prob))
    st.caption(f"Survival probability: {survival_prob:.1f}% | Non-survival probability: {death_prob:.1f}%")

    with st.expander("See the exact input sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Built with scikit-learn + Streamlit as part of the Neurofive ML Track. "
    "Model trained on the classic Titanic dataset (891 passengers)."
)
