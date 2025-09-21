# src/data_preprocessing/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class VANETDataLoader:
    def __init__(self, data_path, rows_limit=None, cache_npz=None, use_cache=True):
        self.data_path = data_path
        self.rows_limit = rows_limit
        self.cache_npz = cache_npz
        self.use_cache = use_cache
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_veremi_data(self):
        """Load VeReMi dataset with optional caching and row limiting."""
        # Load from cache if available
        if self.use_cache and self.cache_npz and os.path.exists(self.cache_npz):
            cached = np.load(self.cache_npz)
            return cached['X'], cached['y']
        # Resolve CSV path
        csv_path = self.data_path
        if os.path.isdir(csv_path):
            csv_path = os.path.join(csv_path, 'veremi_dataset.csv')
        # Read CSV (optionally limited rows)
        if self.rows_limit is not None and self.rows_limit > 0:
            data = pd.read_csv(csv_path, nrows=int(self.rows_limit))
        else:
            data = pd.read_csv(csv_path)
        X, y = self.preprocess_data(data)
        # Save cache
        if self.cache_npz:
            os.makedirs(os.path.dirname(self.cache_npz), exist_ok=True)
            np.savez(self.cache_npz, X=X, y=y)
        return X, y
    
    def preprocess_data(self, data):
        """Basic preprocessing: numeric feature matrix + encoded labels."""
        features = self.extract_behavioral_features(data)
        labels = self.encode_attack_labels(data)
        return features, labels
    
    def extract_behavioral_features(self, data):
        """Extract a numeric feature matrix from the dataframe.
        - Select numeric columns (exclude common label columns if present)
        - Fill missing values with median
        - Standard scale
        """
        df = data.copy()
        # Drop obvious label columns from features
        label_cols = [c for c in ['attack_type', 'label', 'target'] if c in df.columns]
        feature_df = df.drop(columns=label_cols, errors='ignore')
        # Keep only numeric columns
        feature_df = feature_df.select_dtypes(include=[np.number])
        if feature_df.shape[1] == 0:
            # Fall back: attempt to coerce all non-numeric to numeric
            feature_df = df.apply(pd.to_numeric, errors='coerce')
        # Fill missing values
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
        # Scale
        features = self.scaler.fit_transform(feature_df.values)
        return features
    
    def encode_attack_labels(self, data):
        """Encode labels robustly. Prefer 'attack_type', else 'label'/'target'. If none found, return zeros."""
        label_series = None
        for col in ['attack_type', 'label', 'target']:
            if col in data.columns:
                label_series = data[col]
                break
        if label_series is None:
            return np.zeros(len(data), dtype=int)
        return self.label_encoder.fit_transform(label_series.astype(str))
