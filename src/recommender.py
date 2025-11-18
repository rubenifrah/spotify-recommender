import pandas as pd

def get_recommendations(model, scaler, full_df, training_df, feature_cols, top_n=20):
    """
    Predicts liked songs on the unseen portion of the dataset.
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