import unittest
import pandas as pd
import numpy as np
from src.data_processing import balance_dataset

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe
        data = {
            'track_id': [f'id_{i}' for i in range(100)],
            'artists': ['Artist A'] * 50 + ['Artist B'] * 50,
            'liked': [1] * 10 + [0] * 90,
            'track_genre': ['pop'] * 100
        }
        self.df = pd.DataFrame(data)

    def test_balance_dataset_reduces_majority(self):
        """Test that the majority class is undersampled."""
        balanced_df = balance_dataset(self.df, amplification_factor=1, undersample_ratio=2)
        
        liked_count = len(balanced_df[balanced_df['liked'] == 1])
        not_liked_count = len(balanced_df[balanced_df['liked'] == 0])
        
        # We expect roughly 2:1 ratio, but since we have synthetic generation, 
        # liked count might increase slightly.
        # At minimum, not_liked_count should be <= 2 * liked_count
        self.assertLessEqual(not_liked_count, liked_count * 2)
        
    def test_balance_dataset_preserves_liked(self):
        """Test that original liked songs are preserved."""
        original_liked_ids = set(self.df[self.df['liked'] == 1]['track_id'])
        balanced_df = balance_dataset(self.df)
        balanced_liked_ids = set(balanced_df[balanced_df['liked'] == 1]['track_id'])
        
        # Check that all original liked IDs are present in balanced dataset
        # Note: Some might be lost if we had strict undersampling of liked, 
        # but here we only undersample majority.
        # However, the current implementation of balance_dataset keeps all original liked.
        self.assertTrue(original_liked_ids.issubset(balanced_liked_ids))

if __name__ == '__main__':
    unittest.main()
