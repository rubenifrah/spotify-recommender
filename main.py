import os
import pandas as pd
from dotenv import load_dotenv
from src.data_processing import load_and_merge_data, balance_dataset, preprocess_features
from src.models import train_xgboost, evaluate_model
from src.visualization import plot_correlation, plot_pca_2d, plot_xgb_importance, plot_pca_3d_interactive
from src.recommender import get_recommendations
from src.config import (
    KAGGLE_DATA_PATH, 
    MY_PLAYLIST_PATH, 
    RECOMMENDATIONS_PATH, 
    PROCESSED_DATA_DIR
)

def main() -> None:
    # 1. Setup
    load_dotenv()
    # Create dirs if not exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Check files exist
    if not os.path.exists(KAGGLE_DATA_PATH) or not os.path.exists(MY_PLAYLIST_PATH):
        print(f"Error: Please place files in data/raw/")
        print(f"Expected: {KAGGLE_DATA_PATH}")
        print(f"Expected: {MY_PLAYLIST_PATH}")
        return

    # 2. ETL
    print("Loading and processing data...")
    full_df = load_and_merge_data(KAGGLE_DATA_PATH, MY_PLAYLIST_PATH)
    train_df = balance_dataset(full_df)
    
    # 3. Visual Analysis (EDA)
    print("Generating visualizations...")
    plot_correlation(train_df)
    
    # 4. Preprocessing
    print("Preprocessing features...")
    X_train, X_test, y_train, y_test, scaler, feat_cols = preprocess_features(train_df)
    
    # 5. Training
    model = train_xgboost(X_train, y_train)
    
    # 6. Evaluation & Advanced Viz
    evaluate_model(model, X_test, y_test)
    plot_xgb_importance(model)
    plot_pca_2d(X_test, y_test, title="PCA on Test Set")
    plot_pca_3d_interactive(X_test, y_test) # Saves HTML
    
    # 7. Recommendations
    print("Generating recommendations...")
    recs = get_recommendations(model, scaler, full_df, train_df, feat_cols)
    print(f"\nTop Recommendations saved to {RECOMMENDATIONS_PATH}")
    recs[['track_name', 'artists', 'probability']].to_csv(RECOMMENDATIONS_PATH, index=False)

if __name__ == "__main__":
    main()