"""
recommend.py
Core data loading + ML logic for the music recommender.

Design notes (why this differs from a typical "preprocess then load pickle" setup):
- Streamlit Community Cloud only ever runs `streamlit run app.py`. It will NOT run a
  separate preprocessing script for you. So instead of writing .pkl files to disk and
  loading them later, everything here runs in-process and is wrapped in Streamlit's
  caching decorators (in app.py) so it only computes once per app session/restart.
- A full pairwise cosine-similarity matrix (NxN) is memory-heavy — for 10,000 songs
  that's ~800MB, which will crash the free tier's 1GB RAM limit. We use
  scikit-learn's NearestNeighbors on the sparse TF-IDF matrix instead, which finds
  the top-k similar songs without ever materializing an NxN dense matrix.
- No nltk downloads (a network call that can be slow/flaky during a cloud build).
  We use scikit-learn's built-in English stopword list instead.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_PATH = "spotify_millsongdata.csv"

# Cap the number of songs used. Keeps memory/CPU usage safe on free-tier hosting.
# Raise this if you're deploying somewhere with more RAM.
SAMPLE_SIZE = 8000


def _clean_text(text: str) -> str:
    """Lowercase, strip non-letters. Stopword removal is handled by TfidfVectorizer
    itself (stop_words='english'), so this stays cheap and dependency-free."""
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    return text.lower()


def load_data(path: str = DATA_PATH, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["link"], errors="ignore")

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    df = df.reset_index(drop=True)
    df["cleaned_text"] = df["text"].apply(_clean_text)
    return df


def build_model(df: pd.DataFrame, max_features: int = 5000, n_neighbors: int = 11):
    """Returns (vectorizer, tfidf_matrix, nn_model). n_neighbors includes the song
    itself, so it's set to top_n + 1 by the caller."""
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])

    n_neighbors = min(n_neighbors, tfidf_matrix.shape[0])
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    return tfidf, tfidf_matrix, nn_model


def recommend_songs(song_name: str, df: pd.DataFrame, tfidf_matrix, nn_model, top_n: int = 5):
    """Look up a song by (case-insensitive) title and return the top_n most similar
    songs as a DataFrame, or None if the song isn't in the dataset."""
    matches = df.index[df["song"].str.lower() == song_name.lower()]
    if len(matches) == 0:
        return None

    idx = matches[0]
    distances, indices = nn_model.kneighbors(
        tfidf_matrix[idx], n_neighbors=min(top_n + 1, tfidf_matrix.shape[0])
    )

    # Drop the query song itself (always its own nearest neighbor, distance 0)
    result_indices = [i for i in indices[0] if i != idx][:top_n]

    result_df = df[["artist", "song"]].iloc[result_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df
