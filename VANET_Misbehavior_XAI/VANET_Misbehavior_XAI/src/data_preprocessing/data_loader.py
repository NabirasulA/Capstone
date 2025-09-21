# src/data_preprocessing/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class VANETDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_veremi_data(self):
        # Load VeReMi dataset with position falsification attacks
        # Accept either a directory containing 'veremi_dataset.csv' or a direct CSV file path
        csv_path = self.data_path
        if os.path.isdir(csv_path):
            csv_path = os.path.join(csv_path, 'veremi_dataset.csv')
        
        data = pd.read_csv(csv_path)
        return self.preprocess_data(data)
    
    def preprocess_data(self, data):
        # Feature engineering for VANET-specific features
        features = self.extract_behavioral_features(data)
        labels = self.encode_attack_labels(data)
        return features, labels
    
    def extract_behavioral_features(self, data):
        # Extract movement patterns, communication behavior, temporal features
        behavioral_features = []
        # Implementation based on [21][22]
        return np.array(behavioral_features)
    
    def encode_attack_labels(self, data):
        # Encode attack types into numerical labels
        return self.label_encoder.fit_transform(data['attack_type'])
