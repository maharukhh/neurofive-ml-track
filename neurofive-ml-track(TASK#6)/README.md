# 🎵 Spotify Hit Predictor

**Capstone Project — Neurofive ML Track, Week 6**

Predicting whether a song will become a "Hit" on Spotify, using only its
audio characteristics and metadata — the kind of information available
*before* a track is ever released.

---

## 📌 Problem Statement

Every day, thousands of songs are uploaded to streaming platforms, but only
a small fraction ever become genuine hits. Record labels, playlist
curators, and independent artists all face the same question before
committing marketing budget or studio time: **does this track have the
audio DNA of a hit song?**

This project builds a machine learning model that predicts whether a song
is likely to become a **"Hit"** — defined as reaching the top ~10% of
Spotify popularity (popularity score ≥ 70/100) — based purely on its audio
features (danceability, energy, tempo, valence, etc.) and basic metadata
(genre, duration, release date).

**Who this is for:** independent artists and small labels deciding which
tracks to push for marketing or playlist pitching, or producers wanting
quick, data-driven feedback on a track's "hit potential" profile.

## 📊 Dataset

- **Source:** [R4DS TidyTuesday Spotify Songs dataset](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv), originally collected via the Spotify Web API.
- **Size:** 32,833 tracks across 6 genres (pop, rap, rock, latin, r&b, edm), reduced to 26,671 unique, cleanly-dated tracks after preprocessing.
- **Features:** Spotify's own audio-analysis scores (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo), plus genre, duration, and release date.

## 🔧 Approach (Full Workflow)

1. **Data cleaning:** dropped 5 rows missing basic identifiers, deduplicated 4,476 tracks that appeared in multiple playlists (same `track_id`), parsed release dates and dropped 1,681 unparseable ones.
2. **Defined the target:** `is_hit = 1` if `track_popularity >= 70`, else `0` — about **9.5% of tracks are "hits."**
3. **EDA:** compared audio feature distributions between hits and non-hits, genre-level hit rates, and a correlation heatmap — hits skew toward higher danceability, loudness, and valence, and lower acousticness/instrumentalness/speechiness. No single feature dominates, suggesting a multi-feature model is well-suited to this problem.
4. **Feature engineering (4 new features):**
   - `duration_min` — duration in minutes (more interpretable than ms)
   - `song_age_years` — years since release (relative to the Jan 2020 data collection date)
   - `mood_energy` — `valence × energy`, an "upbeat-ness" interaction feature
   - `dance_party` — `danceability` scaled by loudness, a proxy for club/party appeal
5. **Preprocessing pipeline:** `ColumnTransformer` (median imputation + `StandardScaler` for numeric features, most-frequent imputation + `OneHotEncoder` for genre), wrapped in a single `Pipeline` per model to prevent data leakage.
6. **Models trained:** Logistic Regression, Random Forest, and XGBoost — all with class weighting to address the ~90.5%/9.5% class imbalance.
7. **Evaluation:** accuracy, precision/recall/F1 (for the "Hit" class), and ROC-AUC — ROC-AUC used as the primary model-selection metric since it's threshold-independent and more meaningful under class imbalance than accuracy.
8. **Deployment:** the best model was saved with `joblib` and wrapped in a Streamlit web app for live, interactive predictions.

## 📈 Results

| Model                | Accuracy | Precision (Hit) | Recall (Hit) | F1-score (Hit) | ROC-AUC |
|------------------------|:--------:|:------------------:|:---------------:|:-----------------:|:---------:|
| Logistic Regression     | 0.6422   | 0.17                | 0.70             | 0.27               | 0.7425    |
| **Random Forest (best)** | **0.7753** | 0.23              | 0.57             | 0.32               | **0.7695**|
| XGBoost                | 0.7953   | 0.24                | 0.54             | 0.34               | 0.7654    |

**Random Forest was selected as the final model** based on ROC-AUC, narrowly
ahead of XGBoost. It's worth noting this was a close call — XGBoost actually
edged out Random Forest on accuracy, precision, and F1-score, but Random
Forest's better ROC-AUC (its ability to *rank* tracks by hit-likelihood
across all thresholds, not just at one fixed cutoff) made it the more
robust choice for a tool meant to help users compare and prioritize
multiple tracks, not just get a single yes/no answer.

**Top features driving predictions:** `instrumentalness`, `song_age_years`,
genre (`edm`/`pop` flags), `energy`, `loudness`, and the engineered
`dance_party` score all rank in the top features — confirming that both raw
audio characteristics and the engineered interaction features add real
predictive value.

**Honest limitation:** precision for the "Hit" class is modest (0.23) —
meaning most tracks flagged as "hit potential" won't actually become hits.
This is expected and realistic: audio features alone can't capture
marketing spend, artist fame, playlist placement, or cultural timing, all
of which matter enormously in real-world music success. The model is best
used as a **relative screening tool** (which of my 10 demos looks most
promising?) rather than a certainty machine.

## 🚀 Live App

Try it here: https://neurofive-ml-track-task-6.streamlit.app/

## 🖥️ How to Run This Project

### Run the notebook (full analysis)
```bash
pip install -r requirements.txt
jupyter notebook spotify_hit_predictor.ipynb
```

### Run the web app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the app at `http://localhost:8501`.

### Deploy your own copy (Streamlit Community Cloud)
1. Push this folder (`app.py`, `spotify_hit_pipeline.joblib`, `requirements.txt`) to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **"New app"**, select your repo/branch, and point to `app.py`.
4. Click **Deploy** — you'll get a public URL in 1-3 minutes.

## 📁 Project Structure

```
neurofive-ml-track(TASK#6)
├── README.md                          <- you are here
├── CASE_STUDY.md                       <- half-page business case study
├── spotify_hit_predictor.ipynb         <- full analysis notebook (EDA -> modeling -> saving)
├── app.py                              <- Streamlit web app
├── spotify_hit_pipeline.joblib         <- saved, deployable model pipeline
├── requirements.txt                    <- dependencies
├── data/
│   └── spotify_songs.csv               <- raw dataset
└── outputs/                            <- saved charts from the notebook
```

## 🛠️ Tools Used
Python, Pandas, NumPy, scikit-learn (`Pipeline`, `ColumnTransformer`, `RandomForestClassifier`, `LogisticRegression`), XGBoost, Matplotlib, Seaborn, Streamlit, joblib
