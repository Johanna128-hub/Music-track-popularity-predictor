# 🎵 Music Track Popularity Predictor

A machine learning web app that predicts how popular a Spotify track might be, based on its audio features. Enter details like genre, tempo, energy, and danceability to get either a popularity tier classification (Very Low → Very High) or a cluster grouping that shows which type of song yours resembles most.
Built on a dataset of 114,000 Spotify tracks across 114 genres. The classification model uses an ID3 Decision Tree (entropy criterion) and the clustering model uses K-Means++ (k=5, chosen for its production-ready predict() support over Agglomerative Clustering).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Files
- `app.py` — Main Streamlit application
- `id3_model.pkl` — Trained ID3 Decision Tree (criterion=entropy)
- `cluster_model.pkl` — Trained K-Means++ (5 clusters, best silhouette score)
- `scaler.pkl` — StandardScaler for clustering
- `class_labels.pkl` — Classification tier labels
- `features.pkl` — Feature list
- `cluster_stats.pkl` — Cluster popularity statistics

## Models
| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Classification | ID3 (Decision Tree, entropy) | Predicts popularity tier: Very Low → Very High |
| Clustering | K-Means++ (k=5) | Groups track into similar-song clusters |

K-Means++ was chosen after comparing silhouette scores:
- K-Means++: 0.1474 | KMeans random: 0.1473 | Agglomerative: 0.1799 | DBSCAN: 0.0679
- Agglomerative had the best score but lacks a `predict()` method for new data, so K-Means++ (virtually equal score, production-ready) was chosen.
