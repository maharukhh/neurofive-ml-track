"""
Spotify Hit Predictor -- Streamlit Web App
Neurofive ML Track -- Week 6 Capstone Project

Loads the saved Random Forest pipeline (trained on 26,671 Spotify tracks)
and lets a user enter a track's audio features to predict its "Hit"
potential (top ~10% popularity) before release.
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎵", layout="centered")

st.title("🎵 Spotify Hit Predictor")
st.write(
    "Enter a track's audio characteristics below and this model will "
    "estimate its **\"Hit\" potential** -- defined as landing in the top "
    "~10% of Spotify track popularity. It's trained on 26,671 real Spotify "
    "tracks across 6 genres using a Random Forest classifier "
    "(ROC-AUC: **0.77**), wrapped in a full preprocessing pipeline."
)
st.caption(
    "Built for independent artists, producers, and small labels to get a "
    "quick, data-driven read on a track's hit potential before investing "
    "in marketing or playlist pitching."
)

# ---------------------------------------------------------------------------
# Load the saved pipeline (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("spotify_hit_pipeline.joblib")

pipeline = load_pipeline()

st.divider()

# ---------------------------------------------------------------------------
# Input fields for the key audio features
# ---------------------------------------------------------------------------
st.subheader("Track Details")

genre = st.selectbox(
    "Genre",
    options=["pop", "rap", "rock", "latin", "r&b", "edm"],
)

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.65, 0.01,
                              help="How suitable a track is for dancing (0 = least, 1 = most).")
    energy = st.slider("Energy", 0.0, 1.0, 0.65, 0.01,
                        help="Perceptual measure of intensity and activity.")
    valence = st.slider("Valence", 0.0, 1.0, 0.50, 0.01,
                         help="Musical positiveness (0 = sad/angry, 1 = happy/euphoric).")
    loudness = st.slider("Loudness (dB)", -30.0, 0.0, -6.0, 0.5,
                          help="Overall loudness in decibels (typical range: -30 to 0 dB).")
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.15, 0.01)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.08, 0.01,
                             help="Presence of spoken words (higher = more speech-like, e.g. rap/podcasts).")

with col2:
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01,
                                  help="Predicts whether a track has no vocals (closer to 1 = instrumental).")
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01,
                          help="Presence of a live audience in the recording.")
    tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0, 1.0)
    duration_min = st.slider("Duration (minutes)", 1.0, 10.0, 3.5, 0.1)
    release_year = st.number_input("Release Year", min_value=1957, max_value=2020, value=2019)

# ---------------------------------------------------------------------------
# Derived / engineered features (computed automatically, same as training)
# ---------------------------------------------------------------------------
song_age_years = max(0, 2020 - release_year)
mood_energy = valence * energy
dance_party = danceability * (loudness + 60) / 60

st.caption(
    f"Derived automatically: song age = {song_age_years} yrs | "
    f"mood_energy (valence x energy) = {mood_energy:.3f} | "
    f"dance_party score = {dance_party:.3f}"
)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.divider()

if st.button("Predict Hit Potential", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_min": duration_min,
        "song_age_years": song_age_years,
        "mood_energy": mood_energy,
        "dance_party": dance_party,
        "playlist_genre": genre,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]
    hit_prob = probability[1] * 100

    if prediction == 1:
        st.success("### 🔥 Predicted: HIT POTENTIAL")
        st.write(f"Model confidence: **{hit_prob:.1f}%** estimated chance of reaching top-tier popularity")
    else:
        st.info("### 📊 Predicted: NOT LIKELY A BREAKOUT HIT")
        st.write(f"Model confidence: **{hit_prob:.1f}%** estimated chance of reaching top-tier popularity")

    st.progress(int(hit_prob))
    st.caption(
        "Remember: this reflects statistical patterns in audio features only -- "
        "it doesn't account for marketing, artist fame, playlist placement, or "
        "cultural timing, all of which matter enormously in real-world success."
    )

    with st.expander("See the exact input sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Capstone project for the Neurofive ML Track. Model trained on the "
    "TidyTuesday Spotify Songs dataset (32,833 tracks, 6 genres)."
)
