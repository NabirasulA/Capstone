# src/explainability/visualization_utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

def plot_feature_importance(feature_names, importance_values, title="Feature Importance"):
    """Plot feature importance as a bar chart"""
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance_values)[::-1]
    plt.bar(range(len(indices)), importance_values[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix as a heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    return plt.gcf()

def create_interactive_explanation(shap_values, features, feature_names):
    """Create interactive explanation plot using Plotly"""
    # Implementation of interactive visualization
    fig = go.Figure()
    # Add traces based on SHAP values
    return fig

def plot_temporal_explanation(temporal_data, shap_values, feature_idx, feature_name):
    """Plot temporal explanation for time series data"""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(temporal_data[:, feature_idx])
    plt.title(f"Temporal Data for {feature_name}")
    
    plt.subplot(2, 1, 2)
    plt.plot(shap_values[:, feature_idx])
    plt.title(f"SHAP Values for {feature_name}")
    
    plt.tight_layout()
    return plt.gcf()
