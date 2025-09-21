# src/data_preprocessing/feature_extractor.py
import numpy as np
import pandas as pd

class VANETFeatureExtractor:
    def __init__(self, window_size=10):
        self.window_size = window_size
    
    def extract_temporal_features(self, data):
        # Expect data as ndarray or DataFrame with shape (n_samples, n_features)
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        # Create a simple temporal window by repeating each feature vector window_size times
        sequences = np.repeat(X[:, None, :], repeats=self.window_size, axis=1)  # (n_samples, window, n_features)
        # Flatten temporal dimension for downstream reshape in main.py
        temporal_flat = sequences.reshape(n_samples, self.window_size * n_features)
        return temporal_flat
    
    def extract_communication_features(self, data):
        # Placeholder: mirror temporal features for now
        return self.extract_temporal_features(data)
    
    def extract_behavioral_indicators(self, data):
        """Extract behavioral indicators like speed consistency"""
        # Placeholder: basic statistics over features per sample
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        minv = X.min(axis=1, keepdims=True)
        maxv = X.max(axis=1, keepdims=True)
        return np.concatenate([mean, std, minv, maxv], axis=1)
    
    def extract_network_topology(self, data):
        """Extract features related to vehicle density and communication range"""
        # Placeholder: reuse behavioral indicators for shape compatibility
        return self.extract_behavioral_indicators(data)
