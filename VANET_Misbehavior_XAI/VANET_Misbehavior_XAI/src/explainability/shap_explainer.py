# src/explainability/shap_explainer.py
import numpy as np
import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, model_type='deep', sequence_length=None, num_features=None):
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.sequence_length = sequence_length
        self.num_features = num_features
    
    def setup_explainer(self, background_data):
        """Initialize the SHAP explainer with background data"""
        if self.model_type == 'deep':
            # Use KernelExplainer for better TF 2.x compatibility
            try:
                # For sequence models, we need to reshape data for model prediction
                if self.sequence_length and self.num_features:
                    reshaped_background = background_data.reshape(-1, self.sequence_length, self.num_features)
                    self.explainer = shap.DeepExplainer(self.model, reshaped_background)
                else:
                    self.explainer = shap.DeepExplainer(self.model, background_data)
            except Exception:
                # Fallback to KernelExplainer if DeepExplainer fails
                if self.sequence_length and self.num_features:
                    def model_predict(data):
                        reshaped = data.reshape(-1, self.sequence_length, self.num_features)
                        return self.model.predict(reshaped)
                    self.explainer = shap.KernelExplainer(model_predict, background_data)
                else:
                    self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        elif self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'kernel':
            if self.sequence_length and self.num_features:
                def model_predict(data):
                    reshaped = data.reshape(-1, self.sequence_length, self.num_features)
                    return self.model.predict(reshaped)
                self.explainer = shap.KernelExplainer(model_predict, background_data)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
    
    def explain_instance(self, instance):
        """Generate SHAP values for a single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")

        if isinstance(self.explainer, shap.KernelExplainer):
            # KernelExplainer returns a list for multi-class
            shap_values = self.explainer.shap_values(instance)
            if isinstance(shap_values, list):
                return shap_values
            else:
                return [shap_values]
        else:
            # DeepExplainer expects 3D input
            if self.sequence_length and self.num_features:
                reshaped_instance = instance.reshape(-1, self.sequence_length, self.num_features)
            else:
                reshaped_instance = instance
            shap_values = self.explainer.shap_values(reshaped_instance)
            return shap_values
    
    def explain_dataset(self, data):
        """Generate SHAP values for a dataset"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")

        if isinstance(self.explainer, shap.KernelExplainer):
            # KernelExplainer returns a list for multi-class
            shap_values = self.explainer.shap_values(data)
            if isinstance(shap_values, list):
                return shap_values
            else:
                return [shap_values]
        else:
            # DeepExplainer expects 3D input
            if self.sequence_length and self.num_features:
                reshaped_data = data.reshape(-1, self.sequence_length, self.num_features)
            else:
                reshaped_data = data
            shap_values = self.explainer.shap_values(reshaped_data)
            return shap_values
    
    def plot_summary(self, shap_values, features, feature_names=None):
        """Plot SHAP summary plot"""
        try:
            # Handle list format from KernelExplainer
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values_to_plot = shap_values[0]  # Use first class for summary
            else:
                shap_values_to_plot = shap_values

        # Check if we have valid SHAP values
            if shap_values_to_plot is None or len(shap_values_to_plot) == 0:
                print("Warning: No SHAP values to plot")
                return plt.figure()

            # Debug: Print SHAP values shape and some values
            print(f"SHAP values shape: {np.array(shap_values_to_plot).shape}")
            print(f"Features shape: {np.array(features).shape}")
            print(f"Sample SHAP values: {shap_values_to_plot[:3] if len(shap_values_to_plot) > 3 else shap_values_to_plot}")

            # Create feature names if not provided
            if feature_names is None:
                if len(features.shape) > 1:
                    feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
                else:
                    feature_names = [f"Feature_{i}" for i in range(len(features))]

            # Generate summary plot
            shap.summary_plot(shap_values_to_plot, features, feature_names=feature_names, show=False)

            # Ensure the plot is properly created
            fig = plt.gcf()
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            import traceback
            traceback.print_exc()
            return plt.figure()
    
    def plot_dependence(self, shap_values, features, feature_idx, feature_names=None):
        """Plot SHAP dependence plot for a specific feature"""
        # Handle list format from KernelExplainer
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values_to_plot = shap_values[0]  # Use first class for dependence plot
        else:
            shap_values_to_plot = shap_values

        shap.dependence_plot(feature_idx, shap_values_to_plot, features, feature_names=feature_names)
        return plt.gcf()
