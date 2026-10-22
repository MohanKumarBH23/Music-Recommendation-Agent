import os
import joblib
import numpy as np
import pandas as pd

class ContentBasedRecommender:
    """
    Music Recommendation Engine built from scratch using mathematical
    Vector Space Modeling, Min-Max Normalization, and Cosine Similarity.
    
    Operates 100% offline using dataset audio metrics, with zero external APIs.
    """
    FEATURE_COLS = [
        'bpm', 'danceability_%', 'valence_%', 'energy_%',
        'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'
    ]

    # Predefined mood vectors based on acoustic feature distributions
    MOOD_PROFILES = {
        "Happy": {
            "bpm": 125, "danceability_%": 78, "valence_%": 82, "energy_%": 75,
            "acousticness_%": 18, "instrumentalness_%": 0, "liveness_%": 16, "speechiness_%": 6,
            "desc": "Upbeat, positive, high valence songs designed to boost your spirits."
        },
        "Sad": {
            "bpm": 95, "danceability_%": 45, "valence_%": 22, "energy_%": 32,
            "acousticness_%": 72, "instrumentalness_%": 2, "liveness_%": 12, "speechiness_%": 5,
            "desc": "Melancholic, low-energy, introspective ballads with rich acoustic textures."
        },
        "Energetic": {
            "bpm": 138, "danceability_%": 82, "valence_%": 70, "energy_%": 92,
            "acousticness_%": 8, "instrumentalness_%": 1, "liveness_%": 22, "speechiness_%": 12,
            "desc": "High-octane, fast-paced rhythms perfect for workouts, cardio, and gaming."
        },
        "Relaxed / Chill": {
            "bpm": 105, "danceability_%": 58, "valence_%": 48, "energy_%": 38,
            "acousticness_%": 65, "instrumentalness_%": 5, "liveness_%": 11, "speechiness_%": 4,
            "desc": "Calm, ambient, soothing background grooves for relaxing, studying, or winding down."
        },
        "Party": {
            "bpm": 128, "danceability_%": 88, "valence_%": 78, "energy_%": 86,
            "acousticness_%": 10, "instrumentalness_%": 0, "liveness_%": 20, "speechiness_%": 9,
            "desc": "Club bangers with infectious danceability and pounding basslines."
        },
        "Romantic": {
            "bpm": 102, "danceability_%": 60, "valence_%": 55, "energy_%": 45,
            "acousticness_%": 55, "instrumentalness_%": 0, "liveness_%": 14, "speechiness_%": 5,
            "desc": "Warm, heartfelt melodies with balanced tempo and intimate acoustic balance."
        }
    }

    def __init__(self, csv_path='spotify-2023.csv', model_path='spotify_model.pkl'):
        self.csv_path = csv_path
        self.model_path = model_path
        self._load_and_prepare_data()
        self._load_ml_model()

    def _load_and_prepare_data(self):
        """Load, sanitize, and normalize dataset audio features from scratch."""
        self.df = pd.read_csv(self.csv_path, encoding='latin-1')

        # Clean numerical streams
        self.df['streams_cleaned'] = pd.to_numeric(self.df['streams'], errors='coerce')
        median_streams = self.df['streams_cleaned'].median()
        self.df['streams_cleaned'] = self.df['streams_cleaned'].fillna(median_streams).astype(int)

        # Clean release year
        self.df['released_year'] = pd.to_numeric(self.df['released_year'], errors='coerce').fillna(2020).astype(int)

        # Sanitize feature columns
        for col in self.FEATURE_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Build feature matrix
        self.X = self.df[self.FEATURE_COLS].values.astype(float)

        # Min-Max Normalization implemented mathematically from scratch
        self.min_vals = np.min(self.X, axis=0)
        self.max_vals = np.max(self.X, axis=0)
        self.range_denom = np.where((self.max_vals - self.min_vals) == 0, 1.0, self.max_vals - self.min_vals)
        self.X_norm = (self.X - self.min_vals) / self.range_denom

        # Precompute L2 norms for cosine similarity
        self.norms = np.linalg.norm(self.X_norm, axis=1)
        self.norms[self.norms == 0] = 1e-9

    def _load_ml_model(self):
        """Load trained scikit-learn Linear Regression model for stream prediction."""
        if os.path.exists(self.model_path):
            try:
                self.ml_model = joblib.load(self.model_path)
            except Exception:
                self.ml_model = None
        else:
            self.ml_model = None

    def _cosine_similarity(self, target_vector):
        """
        Pure NumPy Cosine Similarity from scratch:
        sim(u, v) = (u . v) / (||u|| * ||v||)
        """
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            target_norm = 1e-9
        dot_products = np.dot(self.X_norm, target_vector)
        return dot_products / (self.norms * target_norm)

    def normalize_vector(self, raw_vector):
        """Project raw feature values into the normalized unit hyperspace [0, 1]."""
        raw_arr = np.array(raw_vector, dtype=float)
        return (raw_arr - self.min_vals) / self.range_denom

    def get_track_names(self):
        """Return a sorted list of unique track titles for UI dropdowns/search."""
        return sorted(self.df['track_name'].dropna().unique().tolist())

    def get_track_recommendations(self, track_name, top_n=10, exclude_same_title=True):
        """
        Find top_n acoustically similar tracks based on cosine similarity
        across all 8 audio features.
        """
        # Search by exact or partial track title
        matches = self.df[self.df['track_name'].str.contains(track_name, case=False, na=False)]
        if matches.empty:
            # Fallback search by artist
            matches = self.df[self.df['artist(s)_name'].str.contains(track_name, case=False, na=False)]
        if matches.empty:
            return None, []

        query_idx = matches.index[0]
        query_song = self.df.iloc[query_idx]
        query_vector = self.X_norm[query_idx]

        sim_scores = self._cosine_similarity(query_vector)
        sorted_indices = np.argsort(sim_scores)[::-1]

        results = []
        seen_titles = set([query_song['track_name'].strip().lower()] if exclude_same_title else [])

        for idx in sorted_indices:
            if idx == query_idx:
                continue
            row = self.df.iloc[idx]
            clean_title = row['track_name'].strip().lower()
            if exclude_same_title and clean_title in seen_titles:
                continue
            seen_titles.add(clean_title)

            results.append({
                'track_name': row['track_name'],
                'artist': row['artist(s)_name'],
                'released_year': int(row['released_year']),
                'streams': f"{row['streams_cleaned']:,}",
                'raw_streams': int(row['streams_cleaned']),
                'similarity': round(float(sim_scores[idx]) * 100, 1),
                'bpm': int(row['bpm']),
                'danceability': int(row['danceability_%']),
                'valence': int(row['valence_%']),
                'energy': int(row['energy_%']),
                'acousticness': int(row['acousticness_%']),
                'speechiness': int(row['speechiness_%']),
                'spotify_search_url': f"https://open.spotify.com/search/{row['track_name']}"
            })
            if len(results) >= top_n:
                break

        return query_song, results

    def get_mood_recommendations(self, mood, top_n=10):
        """
        Recommend songs matching an emotional vibe by calculating similarity
        to the mood's target audio feature profile.
        """
        profile = self.MOOD_PROFILES.get(mood, self.MOOD_PROFILES["Happy"])
        raw_vec = [profile[col] for col in self.FEATURE_COLS]
        norm_vec = self.normalize_vector(raw_vec)

        sim_scores = self._cosine_similarity(norm_vec)
        sorted_indices = np.argsort(sim_scores)[::-1]

        results = []
        seen_titles = set()
        for idx in sorted_indices:
            row = self.df.iloc[idx]
            clean_title = row['track_name'].strip().lower()
            if clean_title in seen_titles:
                continue
            seen_titles.add(clean_title)

            results.append({
                'track_name': row['track_name'],
                'artist': row['artist(s)_name'],
                'released_year': int(row['released_year']),
                'streams': f"{row['streams_cleaned']:,}",
                'raw_streams': int(row['streams_cleaned']),
                'similarity': round(float(sim_scores[idx]) * 100, 1),
                'bpm': int(row['bpm']),
                'danceability': int(row['danceability_%']),
                'valence': int(row['valence_%']),
                'energy': int(row['energy_%']),
                'acousticness': int(row['acousticness_%']),
                'spotify_search_url': f"https://open.spotify.com/search/{row['track_name']}"
            })
            if len(results) >= top_n:
                break
        return results

    def get_custom_vibe_recommendations(self, bpm, danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness, top_n=10):
        """
        Recommend songs matching user-customized acoustic parameters.
        Allows users to synthesize their ideal musical recipe.
        """
        raw_vec = [bpm, danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness]
        norm_vec = self.normalize_vector(raw_vec)

        sim_scores = self._cosine_similarity(norm_vec)
        sorted_indices = np.argsort(sim_scores)[::-1]

        results = []
        seen_titles = set()
        for idx in sorted_indices:
            row = self.df.iloc[idx]
            clean_title = row['track_name'].strip().lower()
            if clean_title in seen_titles:
                continue
            seen_titles.add(clean_title)

            results.append({
                'track_name': row['track_name'],
                'artist': row['artist(s)_name'],
                'released_year': int(row['released_year']),
                'streams': f"{row['streams_cleaned']:,}",
                'raw_streams': int(row['streams_cleaned']),
                'similarity': round(float(sim_scores[idx]) * 100, 1),
                'bpm': int(row['bpm']),
                'danceability': int(row['danceability_%']),
                'valence': int(row['valence_%']),
                'energy': int(row['energy_%']),
                'acousticness': int(row['acousticness_%']),
                'spotify_search_url': f"https://open.spotify.com/search/{row['track_name']}"
            })
            if len(results) >= top_n:
                break
        return results

    def get_top_songs_by_year(self, start_year, end_year, top_n=10):
        """Filter dataset by release year range and return top streamed tracks."""
        filtered = self.df[(self.df['released_year'] >= start_year) & (self.df['released_year'] <= end_year)]
        if filtered.empty:
            filtered = self.df.sort_values(by='streams_cleaned', ascending=False).head(top_n)
        else:
            filtered = filtered.sort_values(by='streams_cleaned', ascending=False).head(top_n)

        results = []
        for _, row in filtered.iterrows():
            results.append({
                'track_name': row['track_name'],
                'artist': row['artist(s)_name'],
                'released_year': int(row['released_year']),
                'streams': f"{row['streams_cleaned']:,}",
                'raw_streams': int(row['streams_cleaned']),
                'danceability': int(row['danceability_%']),
                'energy': int(row['energy_%']),
                'valence': int(row['valence_%']),
                'bpm': int(row['bpm']),
                'spotify_search_url': f"https://open.spotify.com/search/{row['track_name']}"
            })
        return results

    def predict_streams(self, bpm, danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness):
        """Predict total streams using the trained Linear Regression ML model."""
        if self.ml_model is None:
            return None
        try:
            sample_df = pd.DataFrame([[
                bpm, danceability, valence, energy,
                acousticness, instrumentalness, liveness, speechiness
            ]], columns=self.FEATURE_COLS)
            pred = self.ml_model.predict(sample_df)[0]
            return max(0, int(pred))
        except Exception:
            return None
