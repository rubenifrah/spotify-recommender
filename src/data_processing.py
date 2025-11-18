import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import FLAT_GENRE_MAP

def load_and_merge_data(kaggle_path, liked_playlist_path):
    """
    Loads the large Kaggle dataset and the user's liked songs, 
    creates the 'liked' target column.
    """
    df = pd.read_csv(kaggle_path)
    liked_df = pd.read_csv(liked_playlist_path)
    
    # Create set of liked IDs
    liked_ids = set(liked_df['id'].dropna().unique())
    
    # Create Target
    df['liked'] = df['track_id'].apply(lambda x: 1 if x in liked_ids else 0)
    
    # Map Genres
    df['main_genre'] = df['track_genre'].map(FLAT_GENRE_MAP).fillna('other')
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['track_name', 'artists'])
    return df

def balance_dataset(df, amplification_factor=2, undersample_ratio=2):
    """
    1. Oversamples liked songs by finding artists the user likes.
    2. Undersamples non-liked songs to reduce class imbalance.
    """
    # Oversampling Logic (Probabilistic)
    liked_df = df[df['liked'] == 1]
    
    liked_artist_counts = liked_df['artists'].value_counts()
    total_artist_counts = df['artists'].value_counts()
    
    artist_like_ratio = (1/amplification_factor * (liked_artist_counts / total_artist_counts - 1) + 1).fillna(0)
    
    candidates = df[df['liked'] == 0].copy()
    candidates['like_prob'] = candidates['artists'].map(artist_like_ratio).fillna(0)
    
    # Binomial sampling to synthetic "liked"
    np.random.seed(42)
    candidates['synthetic_liked'] = candidates['like_prob'].apply(lambda p: np.random.binomial(1, p))
    
    # Update main dataframe with synthetic likes
    df.loc[candidates[candidates['synthetic_liked'] == 1].index, 'liked'] = 1
    
    # Undersampling Logic
    final_liked = df[df['liked'] == 1]
    final_not_liked = df[df['liked'] == 0]
    
    n_samples = len(final_liked) * undersample_ratio
    not_liked_sampled = final_not_liked.sample(n=n_samples, random_state=42)
    
    balanced_df = pd.concat([final_liked, not_liked_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

def preprocess_features(df):
    """
    One-hot encoding and Scaling.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_columns
    """
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['main_genre'], prefix='genre')
    
    # Define numerical features
    features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 
                'loudness', 'mode', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    
    # Add generated genre columns
    genre_cols = [c for c in df_encoded.columns if c.startswith('genre_')]
    features.extend(genre_cols)
    
    X = df_encoded[features]
    y = df_encoded['liked']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features