# src/explainability/shap_explainer.py
import numpy as np
import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, model_type='deep'):
        self.model = model
        self.model_type = model_type
        self.explainer = None
    
    def setup_explainer(self, background_data):
        """Initialize the SHAP explainer with background data"""
        if self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_data)
        elif self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
    
    def explain_instance(self, instance):
        """Generate SHAP values for a single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")
        
        shap_values = self.explainer.shap_values(instance)
        return shap_values
    
    def explain_dataset(self, data):
        """Generate SHAP values for a dataset"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")
        
        shap_values = self.explainer.shap_values(data)
        return shap_values
    
    def plot_summary(self, shap_values, features, feature_names=None):
        """Plot SHAP summary plot"""
        shap.summary_plot(shap_values, features, feature_names=feature_names)
        return plt.gcf()
    
    def plot_dependence(self, shap_values, features, feature_idx, feature_names=None):
        """Plot SHAP dependence plot for a specific feature"""
        shap.dependence_plot(feature_idx, shap_values, features, feature_names=feature_names)
        return plt.gcf()
