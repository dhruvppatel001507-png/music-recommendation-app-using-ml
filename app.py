import streamlit as st
from recommend import load_data, build_model, recommend_songs

st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",
    layout="centered"
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_resource(show_spinner=False)
def get_model(df):
    return build_model(df)


st.title("🎶 Instant Music Recommender")

with st.spinner("Loading songs..."):
    df = get_data()
    tfidf, tfidf_matrix, nn_model = get_model(df)

song_list = sorted(df["song"].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song, df, tfidf_matrix, nn_model)
        if recommendations is None:
            st.warning("Sorry, song not found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)
