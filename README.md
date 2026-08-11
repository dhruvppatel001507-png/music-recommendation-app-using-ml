# Music Recommendation App using ML

## Overview
This is a **content-based music recommendation system** built with Streamlit, which suggests songs similar to a track selected by the user — based on the actual content/lyrics of songs rather than user listening history or collaborative filtering.

## How it works
- The app is built on top of the **Spotify Million Song Dataset** (`spotify_millsongdata.csv`), which contains song titles, artists, and full lyrics.
- The recommendation logic (`recommend.py`) processes song lyrics — likely applying text vectorization techniques (such as TF-IDF or bag-of-words) followed by a similarity measure (such as cosine similarity) to find songs with the closest lyrical/content match to the one selected.
- The Streamlit frontend (`app.py`) provides a simple UI where a user picks or searches for a song, and the app returns a list of the most similar recommended tracks.

## Key components
- **`app.py`** — Streamlit UI entry point; handles song selection and displays recommendations.
- **`recommend.py`** — core recommendation engine; computes song similarity and returns top matches.
- **`spotify_millsongdata.csv`** — the dataset of songs and lyrics used to power the recommendations.
- **`requirements.txt`** — Python dependencies, likely including `streamlit`, `pandas`, `scikit-learn`, and `nltk` (for text processing).

## Purpose / Use case
A practical demonstration of applying NLP-based content similarity to a real-world recommendation problem — useful as a portfolio project showcasing text processing, vectorization, and building an interactive recommender UI.
