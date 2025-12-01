import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Any
from .config import FLAT_GENRE_MAP

def load_and_merge_data(kaggle_path: str, liked_playlist_path: str) -> pd.DataFrame:
    """
    Loads the large Kaggle dataset and the user's liked songs, 
    creates the 'liked' target column.

    Args:
        kaggle_path: Path to the Kaggle dataset CSV.
        liked_playlist_path: Path to the user's liked songs CSV.

    Returns:
        pd.DataFrame: Merged dataframe with 'liked' column.
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

def balance_dataset(df: pd.DataFrame, amplification_factor: int = 2, undersample_ratio: int = 2) -> pd.DataFrame:
    """
    Balances the dataset using a combination of oversampling and undersampling.

    1. Oversamples liked songs by finding artists the user likes.
    2. Undersamples non-liked songs to reduce class imbalance.

    Args:
        df: Input dataframe.
        amplification_factor: Factor to increase likelihood of selecting songs from liked artists.
        undersample_ratio: Ratio of non-liked to liked songs in the final dataset.

    Returns:
        pd.DataFrame: Balanced dataframe.
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
    # Ensure we don't sample more than available
    n_samples = min(n_samples, len(final_not_liked))
    
    not_liked_sampled = final_not_liked.sample(n=n_samples, random_state=42)
    
    balanced_df = pd.concat([final_liked, not_liked_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, List[str]]:
    """
    One-hot encoding and Scaling.

    Args:
        df: Input dataframe.

    Returns:
        Tuple containing:
            - X_train_scaled (np.ndarray)
            - X_test_scaled (np.ndarray)
            - y_train (pd.Series)
            - y_test (pd.Series)
            - scaler (StandardScaler)
            - feature_columns (List[str])
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