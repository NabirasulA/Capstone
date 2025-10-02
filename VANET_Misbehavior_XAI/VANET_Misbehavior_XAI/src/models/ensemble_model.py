# src/explainability/ensemble_explainer.py
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from .shap_explainer import SHAPExplainer

class EnsembleExplainer:
    def __init__(self, cnn_model, lstm_model, cnn_explainer=None, lstm_explainer=None):
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.ensemble_weights = [0.5, 0.5]  # Equal weights for CNN and LSTM

        # Initialize explainers if not provided
        self.cnn_explainer = cnn_explainer or SHAPExplainer(cnn_model, model_type='kernel')
        self.lstm_explainer = lstm_explainer or SHAPExplainer(lstm_model, model_type='kernel')

    def explain_ensemble_prediction(self, cnn_input, lstm_input, true_label=None):
        """Explain how the ensemble combines CNN and LSTM predictions"""

        # Get individual model predictions
        cnn_pred = self.cnn_model.predict(cnn_input)
        lstm_pred = self.lstm_model.predict(lstm_input)

        # Get individual model SHAP values (flattened for comparison)
        cnn_shap = self.cnn_explainer.explain_instance(cnn_input.reshape(1, -1))
        lstm_shap = self.lstm_explainer.explain_instance(lstm_input.reshape(1, -1))

        # Calculate ensemble prediction
        ensemble_pred = self.ensemble_weights[0] * cnn_pred + self.ensemble_weights[1] * lstm_pred

        # Calculate weighted SHAP values for ensemble
        if isinstance(cnn_shap, list) and len(cnn_shap) > 0:
            cnn_shap_weighted = self.ensemble_weights[0] * cnn_shap[0]
        else:
            cnn_shap_weighted = self.ensemble_weights[0] * cnn_shap

        if isinstance(lstm_shap, list) and len(lstm_shap) > 0:
            lstm_shap_weighted = self.ensemble_weights[1] * lstm_shap[0]
        else:
            lstm_shap_weighted = self.ensemble_weights[1] * lstm_shap

        ensemble_shap = cnn_shap_weighted + lstm_shap_weighted

        return {
            'cnn_prediction': cnn_pred,
            'lstm_prediction': lstm_pred,
            'ensemble_prediction': ensemble_pred,
            'cnn_shap': cnn_shap,
            'lstm_shap': lstm_shap,
            'ensemble_shap': ensemble_shap,
            'true_label': true_label,
            'model_agreement': self._calculate_agreement(cnn_pred, lstm_pred)
        }

    def _calculate_agreement(self, cnn_pred, lstm_pred):
        """Calculate agreement between CNN and LSTM predictions"""
        cnn_class = np.argmax(cnn_pred, axis=1)[0]
        lstm_class = np.argmax(lstm_pred, axis=1)[0]

        return {
            'same_prediction': cnn_class == lstm_class,
            'cnn_confidence': np.max(cnn_pred),
            'lstm_confidence': np.max(lstm_pred),
            'confidence_gap': abs(np.max(cnn_pred) - np.max(lstm_pred))
        }

    def compare_model_explanations(self, cnn_input, lstm_input, feature_names=None):
        """Compare feature importance across CNN, LSTM, and ensemble"""

        # Get explanations for each model
        cnn_result = self.explain_ensemble_prediction(cnn_input, lstm_input)

        # Extract SHAP values for comparison
        cnn_shap = cnn_result['cnn_shap']
        lstm_shap = cnn_result['lstm_shap']
        ensemble_shap = cnn_result['ensemble_shap']

        # Handle list format from SHAP
        if isinstance(cnn_shap, list) and len(cnn_shap) > 0:
            cnn_shap_plot = cnn_shap[0]
        else:
            cnn_shap_plot = cnn_shap

        if isinstance(lstm_shap, list) and len(lstm_shap) > 0:
            lstm_shap_plot = lstm_shap[0]
        else:
            lstm_shap_plot = lstm_shap

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Model predictions comparison
        models = ['CNN', 'LSTM', 'Ensemble']
        predictions = [
            cnn_result['cnn_prediction'].flatten(),
            cnn_result['lstm_prediction'].flatten(),
            cnn_result['ensemble_prediction'].flatten()
        ]

        axes[0, 0].bar(models, [np.max(p) for p in predictions])
        axes[0, 0].set_title('Prediction Confidence by Model')
        axes[0, 0].set_ylabel('Confidence')

        # Feature importance comparison (top 10 features)
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(cnn_shap_plot))]

        # Get top 10 most important features for each model
        cnn_top_features = np.argsort(np.abs(cnn_shap_plot))[-10:]
        lstm_top_features = np.argsort(np.abs(lstm_shap_plot))[-10:]
        ensemble_top_features = np.argsort(np.abs(ensemble_shap))[-10:]

        # Feature overlap analysis
        overlap_cnn_lstm = len(set(cnn_top_features) & set(lstm_top_features))
        overlap_all = len(set(cnn_top_features) & set(lstm_top_features) & set(ensemble_top_features))

        axes[0, 1].bar(['CNN vs LSTM', 'All Models'],
                      [overlap_cnn_lstm, overlap_all])
        axes[0, 1].set_title('Feature Importance Overlap')
        axes[0, 1].set_ylabel('Number of Overlapping Features')

        # Prediction agreement
        agreement = cnn_result['model_agreement']
        axes[1, 0].bar(['Agreement', 'CNN Confidence', 'LSTM Confidence'],
                      [1 if agreement['same_prediction'] else 0,
                       agreement['cnn_confidence'],
                       agreement['lstm_confidence']])
        axes[1, 0].set_title('Model Agreement Analysis')
        axes[1, 0].set_ylabel('Score')

        # Feature importance correlation
        correlation = np.corrcoef(cnn_shap_plot, lstm_shap_plot)[0, 1]
        axes[1, 1].text(0.5, 0.5, f'CNN-LSTM SHAP\nCorrelation: {correlation".3f"}',
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 1].set_title('Model Explanation Correlation')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

        plt.tight_layout()
        return fig, cnn_result

    def explain_ensemble_dataset(self, cnn_data, lstm_data, sample_indices=None):
        """Explain multiple instances from the ensemble"""

        if sample_indices is None:
            sample_indices = np.random.choice(len(cnn_data), min(10, len(cnn_data)), replace=False)

        results = []
        for idx in sample_indices:
            cnn_sample = cnn_data[idx:idx+1]  # Keep batch dimension
            lstm_sample = lstm_data[idx:idx+1]

            result = self.explain_ensemble_prediction(cnn_sample, lstm_sample)
            results.append(result)

        return results

    def plot_ensemble_comparison(self, cnn_data, lstm_data, feature_names=None):
        """Create comprehensive ensemble comparison visualization"""

        # Get sample explanations
        sample_indices = np.random.choice(len(cnn_data), min(5, len(cnn_data)), replace=False)
        explanations = self.explain_ensemble_dataset(cnn_data, lstm_data, sample_indices)

        # Create comparison plot
        fig, axes = plt.subplots(3, len(sample_indices), figsize=(20, 12))

        for i, (sample_idx, explanation) in enumerate(zip(sample_indices, explanations)):
            # CNN SHAP values
            cnn_shap = explanation['cnn_shap']
            if isinstance(cnn_shap, list) and len(cnn_shap) > 0:
                cnn_shap_plot = cnn_shap[0]
            else:
                cnn_shap_plot = cnn_shap

            # LSTM SHAP values
            lstm_shap = explanation['lstm_shap']
            if isinstance(lstm_shap, list) and len(lstm_shap) > 0:
                lstm_shap_plot = lstm_shap[0]
            else:
                lstm_shap_plot = lstm_shap

            # Ensemble SHAP values
            ensemble_shap = explanation['ensemble_shap']

            # Top features for each model
            if feature_names is None:
                feature_names = [f'Feature_{j}' for j in range(len(cnn_shap_plot))]

            # Get top 5 features for each model
            cnn_top = np.argsort(np.abs(cnn_shap_plot))[-5:]
            lstm_top = np.argsort(np.abs(lstm_shap_plot))[-5:]
            ensemble_top = np.argsort(np.abs(ensemble_shap))[-5:]

            # Plot for this sample
            y_pos = np.arange(5)

            # CNN importance
            axes[0, i].barh(y_pos, [abs(cnn_shap_plot[j]) for j in cnn_top])
            axes[0, i].set_yticks(y_pos)
            axes[0, i].set_yticklabels([feature_names[j] for j in cnn_top])
            axes[0, i].set_title(f'Sample {sample_idx}\nCNN Top Features')
            axes[0, i].set_xlabel('Importance')

            # LSTM importance
            axes[1, i].barh(y_pos, [abs(lstm_shap_plot[j]) for j in lstm_top])
            axes[1, i].set_yticks(y_pos)
            axes[1, i].set_yticklabels([feature_names[j] for j in lstm_top])
            axes[1, i].set_title('LSTM Top Features')
            axes[1, i].set_xlabel('Importance')

            # Ensemble importance
            axes[2, i].barh(y_pos, [abs(ensemble_shap[j]) for j in ensemble_top])
            axes[2, i].set_yticks(y_pos)
            axes[2, i].set_yticklabels([feature_names[j] for j in ensemble_top])
            axes[2, i].set_title('Ensemble Top Features')
            axes[2, i].set_xlabel('Importance')

        plt.tight_layout()
        return fig
