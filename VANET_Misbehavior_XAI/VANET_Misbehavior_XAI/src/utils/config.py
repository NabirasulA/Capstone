# src/utils/config.py
import os
import yaml

class Config:
    def __init__(self, config_path=None):
        self.config = {
            # Default configuration values
            'data': {
                'raw_data_path': '../data/raw',
                'processed_data_path': '../data/processed',
                'synthetic_data_path': '../data/synthetic',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'preprocessing': {
                'sequence_length': 10,
                'feature_scaling': True,
                'class_balancing': True
            },
            'model': {
                'cnn': {
                    'filters': [64, 128],
                    'kernel_size': 3,
                    'dense_units': 256,
                    'dropout_rate': 0.3
                },
                'lstm': {
                    'units': [128, 64],
                    'dropout_rate': 0.3
                },
                'ensemble': {
                    'weights': [0.4, 0.4, 0.2]
                }
            },
            'training': {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001,
                'early_stopping_patience': 10
            },
            'explainability': {
                'shap_background_samples': 100,
                'lime_num_features': 10
            }
        }
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
            # Update default config with loaded values
            self._update_dict(self.config, loaded_config)
    
    def save_config(self, config_path):
        """Save current configuration to YAML file"""
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
    
    def _update_dict(self, d, u):
        """Recursively update dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
    
    def get(self, key, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value by key"""
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
