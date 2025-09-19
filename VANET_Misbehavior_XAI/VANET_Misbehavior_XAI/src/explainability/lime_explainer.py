# src/explainability/lime_explainer.py
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

class LIMEExplainer:
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
    
    def setup_explainer(self, training_data):
        """Initialize the LIME explainer with training data"""
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )
    
    def explain_instance(self, instance, num_features=10):
        """Generate LIME explanation for a single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")
        
        explanation = self.explainer.explain_instance(
            instance, 
            self.model.predict_proba,
            num_features=num_features
        )
        return explanation
    
    def plot_explanation(self, explanation, class_idx=1):
        """Plot LIME explanation for a specific class"""
        fig = explanation.as_pyplot_figure(label=class_idx)
        return fig
    
    def get_feature_importance(self, explanation, class_idx=1):
        """Get feature importance from LIME explanation"""
        return explanation.as_list(label=class_idx)
