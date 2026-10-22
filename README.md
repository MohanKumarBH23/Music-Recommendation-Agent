# 🎵 Music Recommendation System & AI Audio Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector_Math-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Engine-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Author: Mohan Kumar](https://img.shields.io/badge/Author-Mohan_Kumar-8A2BE2.svg?logo=github&logoColor=white)](https://github.com/MohanKumarBH23)

A high-performance, dataset-driven **Music Recommendation Engine** and **Interactive Web Application** engineered from scratch with **Vector Space Modeling**, **Min-Max Feature Scaling**, and **Cosine Similarity**.

> ⚡ **Zero External APIs Needed**: Operates 100% offline using dataset audio metrics. No Spotify Developer account, Client ID, Client Secret, or network rate limits required.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Mathematical Architecture](#-mathematical-architecture)
- [Project Structure](#-project-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Application Walkthrough](#-application-walkthrough)
- [Visual Insights & EDA](#-visual-insights--eda)
- [Author & Acknowledgments](#-author--acknowledgments)
- [License](#-license)

---

## 🌟 Overview

Modern music recommendation platforms often rely heavily on complex cloud APIs and strict rate quotas. This project proves that mathematically rigorous, content-based recommendation systems can be implemented **from scratch** to deliver instant, high-accuracy musical discovery.

By analyzing multi-dimensional acoustic metrics (danceability, valence, energy, tempo/BPM, acousticness, speechiness, liveness, and instrumentalness), the engine computes harmonic similarities and suggests personalized recommendations within milliseconds.

---

## ✨ Key Features

- **🔍 Track-Based Recommendations**: Search or choose from 950+ global Spotify hits to inspect acoustic attributes, AI stream forecasts, and the top $N$ closest mathematical matches with percentage similarity scores.
- **🎭 Mood-Based Frequency Matching**: Explore curated emotional profiles (*Happy, Sad, Energetic, Relaxed, Party, Romantic*) mapped directly to acoustic centroid vectors.
- **🎛️ Sonic Vibe Mixer (Audio Synthesizer)**: Build your own custom vibe by dynamically adjusting acoustic sliders (BPM, Energy, Valence, Danceability, Acousticness) to discover songs fitting your exact taste.
- **🤖 AI Stream Predictor**: Built-in trained machine learning model (`spotify_model.pkl`) that estimates global Spotify streams from acoustic feature combinations.
- **📅 Historical Top Songs by Year**: Filter top-streamed songs across decades (1930s to 2023) to observe how audio trends evolved over time.
- **📊 Exploratory Data Analysis (EDA)**: Interactive data visualizations showing streaming distributions, correlation matrices, and artist volume.

---

## 🧠 Mathematical Architecture

The core recommendation engine in [`recommender.py`](recommender.py) is implemented from fundamental linear algebra and vector space modeling principles:

### 1. Acoustic Vector Space Representation
Each track $i$ in the dataset is represented as an 8-dimensional feature vector:

$$\vec{v}_i = \begin{bmatrix} \text{bpm} \\ \text{danceability}\% \\ \text{valence}\% \\ \text{energy}\% \\ \text{acousticness}\% \\ \text{instrumentalness}\% \\ \text{liveness}\% \\ \text{speechiness}\% \end{bmatrix} \in \mathbb{R}^8$$

### 2. Min-Max Normalization (Feature Scaling)
Because features possess vastly different scales (e.g., BPM ranges from 60 to 210, while percentages range from 0 to 100), all dimensions are normalized into the interval $[0, 1]$:

$$\vec{v}_{\text{norm}} = \frac{\vec{v} - \vec{v}_{\min}}{\vec{v}_{\max} - \vec{v}_{\min}}$$

### 3. Cosine Similarity Engine
To determine acoustic proximity independent of vector magnitude, directional cosine similarity is computed:

$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \, \|\vec{v}\|_2} = \frac{\sum_{k=1}^8 u_k v_k}{\sqrt{\sum_{k=1}^8 u_k^2} \sqrt{\sum_{k=1}^8 v_k^2}}$$

This yields an intuitive match score:
$$\text{Match Percentage} = \max(0.0, \text{Cosine Similarity}) \times 100\%$$

### 4. AI Stream Volume Prediction
A multi-variable Linear Regression model predicts estimated lifetime stream counts:

$$\hat{y}_{\text{streams}} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$

---

## 📁 Project Structure

```text
Music-Recommendation-System/
├── app.py                     # Streamlit multi-page web application
├── recommender.py             # Vector space engine, normalization & cosine similarity
├── spotify-2023.csv           # Dataset containing top streamed songs & acoustic features
├── spotify_model.pkl          # Trained machine learning model for stream prediction
├── requirements.txt           # Project dependencies
├── LICENSE                    # MIT License documentation
├── README.md                  # Project documentation
├── Images/                    # Exploratory data analysis charts & plots
│   ├── acousticness_by_key.png
│   ├── danceability_vs_energy.png
│   ├── distribution_of_streams.png
│   ├── top_10_artists.png
│   └── total_streams_by_year.png
└── Notebook/                  # Model training and exploration notebook
    └── spotify_model_creation.ipynb
```

---

## 🚀 Installation & Quickstart

### Prerequisites
- **Python**: Version 3.10 or higher
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/MohanKumarBH23/Music-Recommendation-System.git
cd Music-Recommendation-System
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ Application Walkthrough

| Module | Description |
|---|---|
| **🏠 Home Studio** | Welcome interface, engine health metrics, random song discovery generator, and dataset statistics. |
| **🔍 Track Matcher** | Select any song from the library to inspect its acoustic radar and discover the top similar tracks with precise match percentages. |
| **🎭 Mood Recommender** | Select an emotion (*Happy*, *Sad*, *Energetic*, *Relaxed*, *Party*, *Romantic*) to retrieve tracks aligned with psychological acoustic profiles. |
| **🎛️ Sonic Vibe Mixer** | Customize individual acoustic sliders in real time to query songs that fit your bespoke musical recipe. |
| **📅 Top Songs by Year** | Explore historical hit tracks filtered by custom year ranges (from 1930 to 2023) and streaming rank. |
| **📊 Visual Insights** | In-depth exploratory charts breaking down streams by release decade, top artists, and feature correlations. |

---

## 📈 Visual Insights & EDA

The studio includes curated data science visual artifacts:
- **Total Streams by Release Year**: Visualizes the streaming longevity of modern hits versus timeless catalog tracks.
- **Top 10 Global Artists**: Highlights artists with the highest cumulative global stream volume.
- **Danceability vs. Energy**: Demonstrates acoustic clustering across mood quadrants.
- **Distribution of Streams**: Highlights the power-law distribution typical of viral digital audio hits.
- **Acousticness Across Musical Keys**: Identifies how acoustic density varies with modal key signatures.

---

## 👤 Author

**Mohan Kumar**  
- **GitHub**: [@MohanKumarBH23](https://github.com/MohanKumarBH23)  
- **Portfolio**: [Mohan Kumar's Projects](http://localhost:8080/#projects)  

*Contributions, issues, and feature requests are welcome! Feel free to star ⭐ the repository if you find this project helpful.*

---

## 📄 License

This project is open-source and distributed under the **[MIT License](LICENSE)**.
#   M u s i c - R e c o m m e n d a t i o n - A g e n t  
 