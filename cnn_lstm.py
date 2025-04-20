# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# %%
# Load the preprocessed dataset
df = pd.read_csv("preprocessed_dataset.csv")

# Define features (X) and labels (y)
X = df.drop(columns=['attack'])  # Drop the target column
y = df['attack']

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN input (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# %%
def build_cnn_lstm(input_shape):
    inputs = Input(shape=input_shape)

    # CNN Layer for feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)

    # LSTM Layer for sequential data processing
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    # Fully connected layers (use LSTM output directly)
    x = Flatten()(x)    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification (Benign vs Attack)

    model = Model(inputs, outputs)
    return model

# %%
# Define input shape based on training data
input_shape = (X_train.shape[1], X_train.shape[2])

# Build and compile the model
model = build_cnn_lstm(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Add EarlyStopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save("cnn_lstm_model.h5")

# %%
# CNN Layer:
# Input shape: (None, 20, 1) → 20 timesteps with 1 feature
# Conv1D Layer:
# - Filters: 64
# - Kernel size: 3
# - Activation: ReLU
# - Padding: same
# - Strides: 1
# - Input channels: 1 (from input shape)
# - Output channels: 64 (filters)
# - Output shape: (None, 20, 64) → 64 filters for each of the 20 timesteps
# Params = (kernel_size × input_channels + bias) × filters = (3×1 + 1) × 64 = 256
# Conv1D Layer:
# - Filters: 128
# - Kernel size: 3
# - Activation: ReLU
# - Padding: same
# - Strides: 1
# - Input channels: 64 (from previous layer)
# - Output channels: 128 (filters)
# - Output shape: (None, 20, 128) → 128 filters for each of the 20 timesteps
# Params = (kernel_size × input_channels + bias) × filters = (3×64 + 1) × 128 = 24,704
# Dropout Layer:
# - Dropout rate: 0.3
# - Input shape: (None, 20, 128) → 128 features for each of the 20 timesteps
# - Output shape: (None, 20, 128) → same as input shape
# LSTM Layer:
# - Units: 64
# - Return sequences: True
# - Input shape: (None, 20, 128) → 128 features for each of the 20 timesteps
# - Output shape: (None, 20, 64) → 64 hidden units for each of the 20 timesteps
# Params = 4 × [(input_dim + units) × units + units] = 4 × [(128 + 64) × 64 + 64] = 49,408





