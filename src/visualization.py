import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from xgboost import plot_importance

def plot_correlation(df):
    """Plots correlation matrix of numerical features."""
    plt.figure(figsize=(12, 10))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig('data/processed/correlation_matrix.png')
    print("Saved correlation_matrix.png")

def plot_pca_2d(X_scaled, y, title="PCA 2D Projection"):
    """Projects data to 2D and plots liked vs not-liked."""
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
    df_pca['liked'] = y.values
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_pca, x='PC1', y='PC2', hue='liked', 
        palette={0: 'blue', 1: 'red'}, alpha=0.5
    )
    plt.title(title)
    plt.savefig('data/processed/pca_2d.png')
    print("Saved pca_2d.png")

def plot_pca_3d_interactive(X_scaled, y, genre_labels=None):
    """Generates an HTML interactive 3D plot using Plotly."""
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['liked'] = y.values
    if genre_labels is not None:
        df_pca['genre'] = genre_labels.values
    
    color_col = 'genre' if genre_labels is not None else 'liked'
    
    fig = px.scatter_3d(
        df_pca, x='PC1', y='PC2', z='PC3',
        color=color_col,
        opacity=0.6,
        title="3D PCA Projection"
    )
    fig.write_html("data/processed/pca_3d.html")
    print("Saved pca_3d.html")

def plot_xgb_importance(model, feature_names=None):
    """Plots feature importance from the trained XGBoost model."""
    plt.figure(figsize=(10, 8))
    # We use the built-in plot_importance but manage the axis specifically
    ax = plot_importance(model, max_num_features=20, height=0.5, importance_type='gain', show_values=False)
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png')
    print("Saved feature_importance.png")