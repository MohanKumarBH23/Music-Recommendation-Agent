import os
import streamlit as st
import pandas as pd
import numpy as np
from recommender import ContentBasedRecommender

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spotify-2023.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spotify_model.pkl")
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

DEFAULT_COVER = "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300&auto=format&fit=crop&q=80"

# ---------------------------------------------------------
# Initialize Recommender Engine (Cached)
# ---------------------------------------------------------
@st.cache_resource
def get_recommender():
    return ContentBasedRecommender(csv_path=CSV_PATH, model_path=MODEL_PATH)

engine = get_recommender()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("🎵 Music AI Studio")
st.sidebar.caption("Pure Dataset-Driven Recommender Engine")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Track-Based Recommendations",
        "Mood-Based Recommendations",
        "Sonic Vibe Mixer",
        "Top Songs by Year",
        "Visualizations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Engine Status")
st.sidebar.success(f"🟢 **Dataset Loaded**: {len(engine.df):,} tracks")
if engine.ml_model is not None:
    st.sidebar.success("🤖 **ML Model**: Active (Linear Regression)")
else:
    st.sidebar.warning("⚠️ **ML Model**: Not found")
st.sidebar.info("⚡ **Mode**: 100% Offline & API-Free")

# ---------------------------------------------------------
# 1. Home Page
# ---------------------------------------------------------
if page == "Home":
    st.title("🎵 Music Recommendation System")
    st.markdown("### AI-Powered Music Discovery Built from Scratch (No External APIs)")
    
    col_intro, col_stats = st.columns([2, 1])
    
    with col_intro:
        st.markdown("""
        Welcome to the **Music Recommendation System**! This application uses a **custom-built vector space model** 
        and **mathematical Cosine Similarity** to recommend music based solely on acoustic characteristics 
        extracted from top global tracks.
        
        #### 🧠 How the Recommendation Engine Works:
        1. **Feature Vector Representation**: Each track is modeled as an 8-dimensional vector representing:
           - **Tempo (BPM)**: Cadence and beat frequency.
           - **Danceability**: Rhythm stability and beat strength.
           - **Valence**: Musical positiveness and emotional happiness.
           - **Energy**: Perceptual intensity and activity.
           - **Acousticness & Instrumentalness**: Balance of organic vs synthetic production.
           - **Liveness & Speechiness**: Presence of live audience and spoken words.
        2. **Min-Max Scaling**: Features are normalized from scratch to the $[0, 1]$ interval.
        3. **Cosine Similarity Space**: Measures the angle between normalized feature vectors:
           $$\\text{Cosine Similarity}(\\vec{u}, \\vec{v}) = \\frac{\\vec{u} \\cdot \\vec{v}}{\\|\\vec{u}\\| \\|\\vec{v}\\|}$$
        4. **Stream Predictor Model**: A trained **Linear Regression** model estimates total expected streams based on acoustic parameters.
        """)

    with col_stats:
        st.markdown("#### 📈 Dataset Overview")
        st.metric("Total Catalog Tracks", f"{len(engine.df):,}")
        st.metric("Unique Artists", f"{engine.df['artist(s)_name'].nunique():,}")
        st.metric("Release Year Span", f"{engine.df['released_year'].min()} – {engine.df['released_year'].max()}")
        st.metric("Top Streamed Song", f"{engine.df.sort_values(by='streams_cleaned', ascending=False).iloc[0]['track_name']}")

# ---------------------------------------------------------
# 2. Track-Based Recommendations
# ---------------------------------------------------------
elif page == "Track-Based Recommendations":
    st.title("🔍 Track-Based Recommendations")
    st.write("Pick or search any song from the dataset to find acoustically similar tracks and predict its stream performance.")

    track_list = engine.get_track_names()
    default_index = track_list.index("vampire") if "vampire" in track_list else 0

    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        selected_song = st.selectbox("Select a Song from Catalog:", track_list, index=default_index)
    with col_search2:
        top_k = st.slider("Number of Recommendations:", min_value=3, max_value=20, value=6, step=1)

    query_song, recommendations = engine.get_track_recommendations(selected_song, top_n=top_k)

    if query_song is not None:
        st.markdown("---")
        st.subheader(f"🎶 Selected Track: {query_song['track_name']}")
        
        c_meta, c_ai = st.columns([3, 2])
        with c_meta:
            st.markdown(f"**Artist:** `{query_song['artist(s)_name']}`")
            st.markdown(f"**Release Year:** `{int(query_song['released_year'])}`")
            st.markdown(f"**Actual Streams:** `{int(query_song['streams_cleaned']):,}`")
            st.link_button("🎧 Search & Listen on Spotify", f"https://open.spotify.com/search/{query_song['track_name']}")

        with c_ai:
            st.markdown("#### 🤖 AI Stream Predictor")
            pred_streams = engine.predict_streams(
                bpm=query_song['bpm'],
                danceability=query_song['danceability_%'],
                valence=query_song['valence_%'],
                energy=query_song['energy_%'],
                acousticness=query_song['acousticness_%'],
                instrumentalness=query_song['instrumentalness_%'],
                liveness=query_song['liveness_%'],
                speechiness=query_song['speechiness_%']
            )
            if pred_streams is not None:
                st.metric(label="Predicted Spotify Streams", value=f"{pred_streams:,}")
                st.caption("Generated by the trained Linear Regression model using the song's audio vector.")

        # Audio characteristics metrics
        st.markdown("##### 🎚️ Acoustic Fingerprint")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Tempo", f"{int(query_song['bpm'])} BPM")
        m2.metric("Energy", f"{int(query_song['energy_%'])}%")
        m3.metric("Danceability", f"{int(query_song['danceability_%'])}%")
        m4.metric("Valence (Happiness)", f"{int(query_song['valence_%'])}%")
        m5.metric("Acousticness", f"{int(query_song['acousticness_%'])}%")

        st.markdown("---")
        st.subheader("💡 Most Acoustically Similar Songs (Vector Cosine Distance)")
        
        if recommendations:
            for i, rec in enumerate(recommendations, start=1):
                with st.container():
                    col_num, col_info, col_match, col_btn = st.columns([0.5, 4, 2, 2])
                    with col_num:
                        st.markdown(f"### #{i}")
                    with col_info:
                        st.markdown(f"**{rec['track_name']}**")
                        st.caption(f"Artist: {rec['artist']} | Released: {rec['released_year']}")
                        st.caption(f"Streams: `{rec['streams']}` | Tempo: `{rec['bpm']} BPM`")
                    with col_match:
                        st.metric("Acoustic Match", f"{rec['similarity']}%")
                    with col_btn:
                        st.link_button("🎧 Open Song", rec["spotify_search_url"])
                    st.divider()
        else:
            st.info("No recommendations found.")

# ---------------------------------------------------------
# 3. Mood-Based Recommendations
# ---------------------------------------------------------
elif page == "Mood-Based Recommendations":
    st.title("🎭 Mood-Based Recommendations")
    st.write("Choose your current emotional state to discover songs scientifically matching that acoustic frequency.")

    mood_options = list(engine.MOOD_PROFILES.keys())
    selected_mood = st.pills("Select Your Mood", mood_options, default="Happy")
    
    if selected_mood:
        profile_info = engine.MOOD_PROFILES.get(selected_mood, {})
        st.info(f"**{selected_mood} Profile:** {profile_info.get('desc', '')}")
        
        recs = engine.get_mood_recommendations(selected_mood, top_n=8)
        
        st.subheader(f"✨ Top Tracks for Your '{selected_mood}' Vibe")
        for i, track in enumerate(recs, start=1):
            with st.container():
                c_rank, c_info, c_metric, c_link = st.columns([0.5, 4, 2, 2])
                with c_rank:
                    st.markdown(f"### #{i}")
                with c_info:
                    st.markdown(f"**{track['track_name']}**")
                    st.caption(f"Artist: {track['artist']} | Released: {track['released_year']}")
                    st.caption(f"Energy: `{track['energy']}%` | Valence: `{track['valence']}%` | Danceability: `{track['danceability']}%`")
                with c_metric:
                    st.metric("Mood Match", f"{track['similarity']}%")
                with c_link:
                    st.link_button("🎧 Listen", track["spotify_search_url"])
                st.divider()

# ---------------------------------------------------------
# 4. Sonic Vibe Mixer (Custom Synthesizer)
# ---------------------------------------------------------
elif page == "Sonic Vibe Mixer":
    st.title("🎛️ Sonic Vibe Mixer")
    st.write("Synthesize your own custom acoustic recipe! Adjust the sound parameters below, and the vector model will find the closest songs in existence.")

    c1, c2 = st.columns(2)
    with c1:
        custom_bpm = st.slider("Tempo (BPM)", min_value=60, max_value=200, value=120, step=1)
        custom_energy = st.slider("Energy Level (%)", min_value=0, max_value=100, value=75, step=1)
        custom_danceability = st.slider("Danceability (%)", min_value=0, max_value=100, value=70, step=1)
    with c2:
        custom_valence = st.slider("Valence (Happiness / Positivity %)", min_value=0, max_value=100, value=65, step=1)
        custom_acousticness = st.slider("Acousticness (%)", min_value=0, max_value=100, value=25, step=1)
        custom_speechiness = st.slider("Speechiness (%)", min_value=0, max_value=100, value=10, step=1)

    custom_recs = engine.get_custom_vibe_recommendations(
        bpm=custom_bpm,
        danceability=custom_danceability,
        valence=custom_valence,
        energy=custom_energy,
        acousticness=custom_acousticness,
        instrumentalness=0,
        liveness=15,
        speechiness=custom_speechiness,
        top_n=8
    )

    st.markdown("---")
    st.subheader("🎶 Closest Matching Songs to Your Custom Recipe")
    
    for i, track in enumerate(custom_recs, start=1):
        with st.container():
            c_rank, c_info, c_metric, c_link = st.columns([0.5, 4, 2, 2])
            with c_rank:
                st.markdown(f"### #{i}")
            with c_info:
                st.markdown(f"**{track['track_name']}**")
                st.caption(f"Artist: {track['artist']} | Released: {track['released_year']}")
                st.caption(f"BPM: `{track['bpm']}` | Energy: `{track['energy']}%` | Valence: `{track['valence']}%` | Dance: `{track['danceability']}%`")
            with c_metric:
                st.metric("Vibe Match", f"{track['similarity']}%")
            with c_link:
                st.link_button("🎧 Listen", track["spotify_search_url"])
            st.divider()

# ---------------------------------------------------------
# 5. Top Songs by Year
# ---------------------------------------------------------
elif page == "Top Songs by Year":
    st.title("📅 Top Songs by Year Range")
    st.write("Explore top streamed hits filtered across historical release years.")

    start_year, end_year = st.slider(
        "Select Year Range:",
        min_value=int(engine.df['released_year'].min()),
        max_value=int(engine.df['released_year'].max()),
        value=(2018, 2023),
        step=1
    )

    top_tracks = engine.get_top_songs_by_year(start_year, end_year, top_n=10)

    st.subheader(f"🏆 Top Streamed Tracks ({start_year} – {end_year})")
    for i, track in enumerate(top_tracks, start=1):
        with st.container():
            col_rank, col_txt, col_streams, col_btn = st.columns([0.5, 4, 2, 2])
            with col_rank:
                st.markdown(f"### #{i}")
            with col_txt:
                st.markdown(f"**{track['track_name']}**")
                st.caption(f"Artist: {track['artist']} | Year: {track['released_year']}")
                st.caption(f"Danceability: `{track['danceability']}%` | Energy: `{track['energy']}%` | Valence: `{track['valence']}%`")
            with col_streams:
                st.metric("Total Streams", track['streams'])
            with col_btn:
                st.link_button("🎧 Search Spotify", track["spotify_search_url"])
            st.divider()

# ---------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------
elif page == "Visualizations":
    st.title("📊 Exploratory Data Analysis & Visual Insights")
    st.markdown("These visual insights were generated through data science workflows on Spotify's top streamed songs.")

    viz_list = [
        ("total_streams_by_year.png", "📈 Total Streams by Release Year", "Shows how modern releases dominate overall streaming volume, while evergreen classics sustain enduring popularity."),
        ("top_10_artists.png", "🎤 Top 10 Artists by Total Streams", "Highlights the most-streamed global superstars dominating global playlists."),
        ("danceability_vs_energy.png", "⚡ Danceability vs Energy", "A scatter plot exploring the relationship between danceable grooves and high energy tracks."),
        ("distribution_of_streams.png", "📊 Distribution of Streams", "Histogram and density estimation showing the power-law distribution of global music streams."),
        ("acousticness_by_key.png", "🎹 Acousticness Across Musical Keys", "Breakdown of acoustic qualities depending on the key signature and modality of tracks.")
    ]

    for filename, caption, description in viz_list:
        img_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(img_path):
            st.subheader(caption)
            try:
                st.image(img_path, caption=caption, width="stretch")
            except TypeError:
                st.image(img_path, caption=caption)
            st.caption(description)
            st.divider()
        else:
            st.warning(f"Image '{filename}' not found in {IMAGES_DIR}.")
