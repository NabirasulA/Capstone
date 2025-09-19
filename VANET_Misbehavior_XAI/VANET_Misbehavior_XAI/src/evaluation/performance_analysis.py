# src/evaluation/performance_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_roc_curve(y_true, y_prob, class_names=None):
    """Plot ROC curve for multi-class classification"""
    n_classes = y_prob.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        class_label = class_names[i] if class_names else f"Class {i}"
        plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_prob, class_names=None):
    """Plot Precision-Recall curve for multi-class classification"""
    n_classes = y_prob.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        pr_auc = auc(recall, precision)
        
        class_label = class_names[i] if class_names else f"Class {i}"
        plt.plot(recall, precision, lw=2, label=f'{class_label} (AUC = {pr_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    return plt.gcf()

def analyze_misclassifications(y_true, y_pred, X, feature_names=None):
    """Analyze misclassified instances"""
    misclassified_idx = np.where(y_true != y_pred)[0]
    misclassified_data = X[misclassified_idx]
    misclassified_true = y_true[misclassified_idx]
    misclassified_pred = y_pred[misclassified_idx]
    
    misclassification_df = pd.DataFrame(misclassified_data)
    if feature_names:
        misclassification_df.columns = feature_names
    
    misclassification_df['true_label'] = misclassified_true
    misclassification_df['predicted_label'] = misclassified_pred
    
    return misclassification_df
