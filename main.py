import os
import pandas as pd
from dotenv import load_dotenv
from src.data_processing import load_and_merge_data, balance_dataset, preprocess_features
from src.models import train_xgboost, evaluate_model
from src.visualization import plot_correlation, plot_pca_2d, plot_xgb_importance, plot_pca_3d_interactive
from src.recommender import get_recommendations

def main():
    # 1. Setup
    load_dotenv()
    # Create dirs if not exist
    os.makedirs('data/processed', exist_ok=True)
    
    KAGGLE_DATA = 'data/raw/spotify_songs_with_audio_features.csv'
    MY_PLAYLIST = 'data/raw/all_my_songs.csv'
    
    # Check files exist
    if not os.path.exists(KAGGLE_DATA) or not os.path.exists(MY_PLAYLIST):
        print("Error: Please place 'spotify_songs_with_audio_features.csv' and 'all_my_songs.csv' in data/raw/")
        return

    # 2. ETL
    full_df = load_and_merge_data(KAGGLE_DATA, MY_PLAYLIST)
    train_df = balance_dataset(full_df)
    
    # 3. Visual Analysis (EDA)
    print("Generating visualizations...")
    plot_correlation(train_df)
    
    # 4. Preprocessing
    X_train, X_test, y_train, y_test, scaler, feat_cols = preprocess_features(train_df)
    
    # 5. Training
    model = train_xgboost(X_train, y_train)
    
    # 6. Evaluation & Advanced Viz
    evaluate_model(model, X_test, y_test)
    plot_xgb_importance(model)
    plot_pca_2d(X_test, y_test, title="PCA on Test Set")
    plot_pca_3d_interactive(X_test, y_test) # Saves HTML
    
    # 7. Recommendations
    recs = get_recommendations(model, scaler, full_df, train_df, feat_cols)
    print(f"\nTop Recommendations saved to data/processed/recommendations.csv")
    recs[['track_name', 'artists', 'probability']].to_csv('data/processed/recommendations.csv', index=False)

if __name__ == "__main__":
    main()