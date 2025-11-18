
# 🎶 Spotify recommender : a personalized music taste modeling

This repository contains a complete, modular, production‑grade pipeline that learns **your personal musical taste** using:
- A **large Kaggle Spotify dataset** containing audio features for tens of thousands of tracks  
- Your own **liked songs** exported from Spotify  

The project trains a classifier (XGBoost) to predict the probability that you will like a new song based on audio features, genre embeddings, and historical preferences. It also produces visualizations, PCA embeddings, feature importance plots, balanced datasets, and a fully reproducible recommendation pipeline.

For a more in-depth look at the project, please check the following Google Collab: [Spotify recommender colab](https://colab.research.google.com/drive/1YhSFuuFKE66bT0vQ3XHLxExes16yGWG5?usp=sharing)

---

# 📖 Project summary

Most modern recommendation systems rely on **collaborative filtering**, which works well for popular content but is inherently conservative:

- They recommend songs *similar to what everyone else likes*
- They bias toward mainstream tracks
- They rarely surface *new*, *unheard*, or *idiosyncratically relevant* songs

This project breaks out of that paradigm.

Instead of relying on global patterns, I train a **supervised binary classifier** uniquely on **my personal taste**, improving:

### 🎯 Personalization  
The learned decision boundary reflects *my* musical preferences, not the crowd’s.

### 🎯 Discovery  
The model identifies songs I am likely to love **but have never heard**, enabling genuine exploration.

### 🎯 Interpretability  
The modeling pipeline reveals whether my taste is driven by:
- acoustic properties (valence, tempo, energy)  
- cultural/contextual factors (genre, popularity, release year)  

---

# 🏛️ Repository architecture

The project follows a **modular, production‑grade Python package structure**, ensuring scalability, readability, and clean separation of concerns.

```
spotify-recommender/
├── data/
│   ├── raw/                   # Kaggle dataset + 'all_my_songs.csv'
│   └── processed/             # Recommendations, figures, PCA HTML
├── src/
│   ├── __init__.py
│   ├── config.py              # Genre maps, API keys, model config
│   ├── data_processing.py     # ETL, balancing, feature engineering
│   ├── models.py              # Training + evaluation (XGBoost, MLP)
│   ├── recommender.py         # Final scoring + ranking
│   ├── spotify_client.py      # Spotify API wrapper
│   └── visualization.py       # PCA plots, importance plots, correlations
├── main.py                    # Full execution pipeline
└── requirements.txt
```

The `src/` directory is structured like a **real ML codebase** rather than a monolithic notebook.

---

# ⚙️ Methodology and pipeline deep dive

The project is built around a **full ML pipeline**, from raw data to evaluation to recommendation generation.

---

# 🔧 Data engineering & custom balancing  
_Implemented in `data_processing.py`_

## 🔹 A. Dataset Merge & Target Creation

Two sources are merged:

1. **Kaggle dataset:**  
   Contains tens of thousands of tracks with:
   - acoustic features (danceability, valence, energy…)
   - metadata (artists, genres, release year)
   - popularity metrics

2. **Personal liked songs (`all_my_songs.csv`):**  
   Contains ~1899 tracks I personally liked.

A binary target is created:

```
liked = 1  if track_id ∈ my liked songs  
liked = 0  otherwise
```

## 🔹 B. Feature engineering

Spotify’s ultra‑granular `track_genre` field (hundreds of values) is condensed into **10 macrogenres**:

- pop  
- rock  
- electronic  
- hiphop_rnb  
- jazz_blues  
- latin  
- world  
- folk_country  
- classical  
- other  

This improves:
- generalization  
- interpretability  
- reduces sparsity  
- avoids overfitting on niche genres  

## 🔹 C. Custom balancing strategy (critical step)

The raw dataset is **severely imbalanced**:
- Liked songs: **~1,900**
- Not‑liked songs: **>150,000+**

Left untreated, the classifier would always predict 0.

### The solution combines:

#### - **Oversampling**
Synthetic “liked” examples are created by:  
- finding non‑liked songs by frequently liked artists  
- selecting acoustically similar tracks  
- probabilistically flipping their label  

This enriches the minority class with musically coherent examples.

#### - **Undersampling**
The majority class is randomly reduced to produce a **2:1 ratio** of not‑liked to liked.

This ensures the classifier focuses on *signal*, not *noise*.

---

# 🤖 Modeling & evaluation  
_Implemented in `models.py`_

Multiple models were evaluated:

| Model | Strengths | Weaknesses |
|-------|-----------|-------------|
| **XGBoost (Selected)** | Best F1, robust, interpretable | Requires tuning |
| MLP | Captures nonlinearities | Less stable |
| Random Forest | Interpretable | Lower recall |

---

# 🔍 Hyperparameter runing

A GridSearchCV with **5‑fold cross‑validation** was run on:
- tree depth  
- learning rate  
- gamma  
- subsampling  
- regularization terms

Optimization metric:

```
maximize F1-score on the minority class (liked=1)
```

Because missing a potential favorite song is costly.

---

# 📈 Final model results (real rxtracted metrics)

These results come directly from your notebook.

## **Confusion matrix**
```
[[678  28]
 [ 43 226]]
```

## **Classification report**

| Class | Precision | Recall | F1‑Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Not Liked) | 0.94 | 0.96 | 0.95 | 706 |
| 1 (Liked) | 0.89 | 0.84 | 0.86 | 269 |

### **Weighted F1: 0.93**  
### **Accuracy: 0.93**

This is exceptionally strong given the imbalance and personal‑taste variability.

---

# 🔎 Key insights from model performance

### - High recall on liked songs (0.84)  
→ The model successfully captures the essence of my taste.

### - High precision on liked songs (0.89)  
→ False positives (irrelevant recommendations) are low.

### - Balanced, stable classification boundary  
→ Model does not collapse into trivial majority-class predictions.

---

# 📈 Interpretation & visualization  
_Implemented in `visualization.py`_

## 🧮 Feature importance

The model reveals **what truly defines my taste**:

| Feature Category | Top Predictors | Interpretation |
|------------------|----------------|----------------|
| **Contextual / Cultural** | popularity, release_year, main_genre_pop | My preferences are shaped by era, mainstream patterns, and genre |
| **Audio Features** | valence, tempo, loudness | Plays a role, but secondary |

### 🧩 Storytelling Insight  
Contrary to the idea that taste is purely sonic, my preferences are strongly tied to **context, culture, and era**.

---

## 📡 PCA embeddings (2D & interactive 3D)

The PCA projections reveal:

- Liked songs cluster in **two major regions**:
  1. energetic, upbeat, rhythmic  
  2. calm, low‑valence, atmospheric  
- Non‑liked songs are widely dispersed  
- A clear linear separation appears in PC1–PC2 space

Interactive version saved as:

```
data/processed/pca_3d.html
```

---

# 🎧 Recommendations

The pipeline ranks all unseen tracks using:

```
P(liked = 1 | features)
```

This produces a playlist of **100 recommendations**, which was exported and uploaded:

Example playlist ID from notebook:  
```
5uKn4ly10ZHqSlG4N1xZ0U
```

Insights:
- High proportion of new discoveries  
- Strong genre diversity  
- Intelligent balancing between familiar artists & novel suggestions  

---

# 🚀 Getting started

## 1. Installation

```bash
git clone https://github.com/yourusername/spotify-recommender.git
cd spotify-recommender
pip install -r requirements.txt
```

## 2. Configure credentials

Create a `.env` file:

```
SPOTIFY_CLIENT_ID="your_id"
SPOTIFY_CLIENT_SECRET="your_secret"
SPOTIFY_USER_ID="your_user"
```

## 3. Add data

Place in `data/raw/`:
- `spotify_songs_with_audio_features.csv`
- `all_my_songs.csv`

## 4. Run the pipeline

```
python main.py
```

---

# ⚠️ Limitations

### 1. Data coverage bias  
Some of my liked songs (especially non‑mainstream French tracks) were **missing in the Kaggle dataset**, creating blind spots.

### 2. Cold‑start problem  
Completely novel genres or artists remain hard to classify.

---

# 🛠️ Future work

### -> Full Spotify API ingestion  
Replace Kaggle dataset with complete, up‑to‑date personal audio feature scraping.

### -> Ensemble models  
Combine XGBoost (high recall) + MLP (generalization power).

### -> Lyrics/NLP integration  
Use transformer embeddings to incorporate thematic and emotional features.



---

# 👤 Author  
📌 **Ruben Ifrah**  
[LinkedIn](https://www.linkedin.com/in/ruben-ifrah-245496253/)
 
