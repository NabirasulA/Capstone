# src/models/cnn_model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class VANETCNNModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # 1D CNN for sequential behavior patterns
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalMaxPooling1D()(x)
        
        # Dense layers for classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name="vanet_cnn")
        return model
