# src/models/lstm_model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class VANETLSTMModel:
    def __init__(self, sequence_length, num_features, num_classes):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
    
    def build_model(self):
        inputs = tf.keras.Input(shape=(self.sequence_length, self.num_features))
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name="vanet_lstm")
        return model
