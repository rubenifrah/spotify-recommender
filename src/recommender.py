import pandas as pd
from typing import List, Any
from sklearn.preprocessing import StandardScaler

def get_recommendations(model: Any, scaler: StandardScaler, full_df: pd.DataFrame, training_df: pd.DataFrame, feature_cols: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Predicts liked songs on the unseen portion of the dataset.

    Args:
        model: Trained model (XGBoost or MLP).
        scaler: Fitted StandardScaler.
        full_df: Complete dataframe containing all songs.
        training_df: Dataframe used for training (to exclude seen songs).
        feature_cols: List of feature column names used for training.
        top_n: Number of recommendations to return.

    Returns:
        pd.DataFrame: Top N recommended songs with 'probability' column.
    """
    # Identify unseen songs (songs in full_df but not in training_df)
    seen_ids = set(training_df['track_id'])
    unseen_df = full_df[~full_df['track_id'].isin(seen_ids)].copy()
    
    # Preprocess unseen data exactly like training data
    unseen_encoded = pd.get_dummies(unseen_df, columns=['main_genre'], prefix='genre')
    
    # Ensure all columns exist (fill missing genre cols with 0)
    for col in feature_cols:
        if col not in unseen_encoded.columns:
            unseen_encoded[col] = 0
            
    X_unseen = unseen_encoded[feature_cols]
    X_unseen_scaled = scaler.transform(X_unseen)
    
    # Predict probabilities
    probs = model.predict_proba(X_unseen_scaled)[:, 1]
    unseen_df['probability'] = probs
    
    # Return top N
    return unseen_df.sort_values(by='probability', ascending=False).head(top_n)