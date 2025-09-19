# src/models/model_utils.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    """Create sequences for time series models"""
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
    return np.array(xs)

def train_test_split_temporal(data, labels, train_ratio=0.7, val_ratio=0.15):
    """Split data preserving temporal order"""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data, train_labels = data[:train_end], labels[:train_end]
    val_data, val_labels = data[train_end:val_end], labels[train_end:val_end]
    test_data, test_labels = data[val_end:], labels[val_end:]
    
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    return plt
