# main.py - Main entry point for VANET Misbehavior Detection with XAI
import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing.data_loader import VANETDataLoader
from src.data_preprocessing.feature_extractor import VANETFeatureExtractor
from src.models.cnn_model import VANETCNNModel
from src.models.lstm_model import VANETLSTMModel
from src.models.ensemble_model import XAIEnsembleVANET
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.evaluation.metrics import calculate_classification_metrics
from src.utils.config import Config
from src.utils.logging_utils import setup_logger, Timer

def setup_gpu(logger):
    """Enable GPU memory growth and report available devices."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            logger.info(f"GPUs available: {gpus}")
        else:
            logger.info("No GPU detected; running on CPU.")
    except Exception as e:
        logger.warning(f"GPU setup encountered an issue: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='VANET Misbehavior Detection with XAI')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None, help='Path to raw data directory or CSV file (defaults to config value)')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'explain'], default='train',
                        help='Operation mode')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model (for evaluate/explain mode)')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--rows_limit', type=int, default=None, help='Only load the first N rows from the CSV (for quick tests)')
    parser.add_argument('--cache_npz', type=str, default=None, help='Path to a .npz cache file to speed up subsequent loads')
    return parser.parse_args()

def train(config, data_path, logger, rows_limit=None, cache_npz=None):
    logger.info("Starting training process")
    
    # Load and preprocess data
    with Timer("Data loading"):
        effective_rows_limit = rows_limit if rows_limit is not None else config.get('data.rows_limit')
        data_loader = VANETDataLoader(
            data_path,
            rows_limit=effective_rows_limit,
            cache_npz=cache_npz,
            use_cache=True
        )
        X, y = data_loader.load_veremi_data()
    
    # Feature extraction
    with Timer("Feature extraction"):
        feature_extractor = VANETFeatureExtractor(window_size=config.get('preprocessing.sequence_length'))
        temporal_features = feature_extractor.extract_temporal_features(X)
        communication_features = feature_extractor.extract_communication_features(X)
        # Debug shapes
        logger.info(f"Raw features shape: {np.asarray(X).shape}")
        logger.info(f"Temporal features shape: {np.asarray(temporal_features).shape}")
        logger.info(f"Communication features shape: {np.asarray(communication_features).shape}")
        # Defensive fixes
        if temporal_features is None or np.asarray(temporal_features).ndim == 0:
            logger.error("Temporal features are empty. Please verify dataset columns.")
            return
        temporal_features = np.asarray(temporal_features)
        if temporal_features.ndim == 1:
            temporal_features = temporal_features.reshape(-1, 1)
    
    # Prepare data for different models
    sequence_length = config.get('preprocessing.sequence_length')
    if temporal_features.shape[1] == 0:
        logger.error("No feature columns after feature extraction. Aborting.")
        return
    num_features = temporal_features.shape[1] // sequence_length if temporal_features.shape[1] >= sequence_length else temporal_features.shape[1]
    num_classes = len(np.unique(y))
    
    # Reshape data for CNN and LSTM models
    X_cnn = temporal_features.reshape(-1, sequence_length, num_features)
    X_lstm = X_cnn.copy()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train_cnn, X_test_cnn, X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(
        X_cnn, X_lstm, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Build models
    with Timer("Model building"):
        # CNN model
        cnn_model = VANETCNNModel(input_shape=(sequence_length, num_features), num_classes=num_classes)
        cnn = cnn_model.build_model()
        cnn.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('training.learning_rate')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # LSTM model
        lstm_model = VANETLSTMModel(sequence_length=sequence_length, num_features=num_features, num_classes=num_classes)
        lstm = lstm_model.build_model()
        lstm.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('training.learning_rate')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Train models
    os.makedirs('results/models', exist_ok=True)
    with Timer("CNN training"):
        try:
            cnn_history = cnn.fit(
                X_train_cnn, y_train,
                batch_size=config.get('training.batch_size'),
                epochs=config.get('training.epochs'),
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=config.get('training.early_stopping_patience'),
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath='results/models/cnn_epoch_{epoch:02d}.weights.h5',
                        save_weights_only=True,
                        save_freq='epoch'
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath='results/models/cnn_best.weights.h5',
                        save_weights_only=True,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True
                    )
                ]
            )
        except KeyboardInterrupt:
            logger.warning("CNN training interrupted. Saving current weights.")
            cnn.save_weights('results/models/cnn_interrupted.weights.h5')
            return
        
    with Timer("LSTM training"):
        try:
            lstm_history = lstm.fit(
                X_train_lstm, y_train,
                batch_size=config.get('training.batch_size'),
                epochs=config.get('training.epochs'),
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=config.get('training.early_stopping_patience'),
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath='results/models/lstm_epoch_{epoch:02d}.weights.h5',
                        save_weights_only=True,
                        save_freq='epoch'
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath='results/models/lstm_best.weights.h5',
                        save_weights_only=True,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True
                    )
                ]
            )
        except KeyboardInterrupt:
            logger.warning("LSTM training interrupted. Saving current weights.")
            lstm.save_weights('results/models/lstm_interrupted.weights.h5')
            return
    
        # Save models (Keras 3 requires explicit extension)
    os.makedirs('results/models', exist_ok=True)
    cnn.save('results/models/cnn_model.keras')
    lstm.save('results/models/lstm_model.keras')
    
    # Evaluate models
    cnn_pred = cnn.predict(X_test_cnn)
    lstm_pred = lstm.predict(X_test_lstm)
    
    cnn_pred_classes = np.argmax(cnn_pred, axis=1)
    lstm_pred_classes = np.argmax(lstm_pred, axis=1)
    
    cnn_metrics = calculate_classification_metrics(y_test, cnn_pred_classes, cnn_pred)
    lstm_metrics = calculate_classification_metrics(y_test, lstm_pred_classes, lstm_pred)
    
    logger.info(f"CNN Model Metrics: {cnn_metrics}")
    logger.info(f"LSTM Model Metrics: {lstm_metrics}")
    
    # Create ensemble model
    ensemble = XAIEnsembleVANET(cnn, lstm)
    ensemble_pred = ensemble.predict(X_test_cnn, X_test_lstm)
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    
    ensemble_metrics = calculate_classification_metrics(y_test, ensemble_pred_classes, ensemble_pred)
    logger.info(f"Ensemble Model Metrics: {ensemble_metrics}")
    
    return cnn, lstm, ensemble

def evaluate(config, model_path, data_path, logger, rows_limit=None, cache_npz=None):
    logger.info(f"Evaluating model from {model_path}")
    
    # Load data
    effective_rows_limit = rows_limit if rows_limit is not None else config.get('data.rows_limit')
    data_loader = VANETDataLoader(
        data_path,
        rows_limit=effective_rows_limit,
        cache_npz=cache_npz,
        use_cache=True
    )
    X, y = data_loader.load_veremi_data()
    
    # Feature extraction (same as in train)
    with Timer("Feature extraction"):
        feature_extractor = VANETFeatureExtractor(window_size=config.get('preprocessing.sequence_length'))
        temporal_features = feature_extractor.extract_temporal_features(X)
        # Defensive fixes
        if temporal_features is None or np.asarray(temporal_features).ndim == 0:
            logger.error("Temporal features are empty. Please verify dataset columns.")
            return
        temporal_features = np.asarray(temporal_features)
        if temporal_features.ndim == 1:
            temporal_features = temporal_features.reshape(-1, 1)
    
    # Reshape for model input (same as in train)
    sequence_length = config.get('preprocessing.sequence_length')
    if temporal_features.shape[1] == 0:
        logger.error("No feature columns after feature extraction. Aborting.")
        return
    num_features = temporal_features.shape[1] // sequence_length if temporal_features.shape[1] >= sequence_length else temporal_features.shape[1]
    X_model = temporal_features.reshape(-1, sequence_length, num_features)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Evaluate
    y_pred = model.predict(X_model)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    metrics = calculate_classification_metrics(y, y_pred_classes, y_pred)
    logger.info(f"Model Metrics: {metrics}")
    
    return metrics

def explain(config, model_path, data_path, logger, rows_limit=None, cache_npz=None):
    logger.info(f"Generating explanations for model from {model_path}")
    
    # Load data
    effective_rows_limit = rows_limit if rows_limit is not None else config.get('data.rows_limit')
    data_loader = VANETDataLoader(
        data_path,
        rows_limit=effective_rows_limit,
        cache_npz=cache_npz,
        use_cache=True
    )
    X, y = data_loader.load_veremi_data()
    
    # Feature extraction (same as in train)
    with Timer("Feature extraction"):
        feature_extractor = VANETFeatureExtractor(window_size=config.get('preprocessing.sequence_length'))
        temporal_features = feature_extractor.extract_temporal_features(X)
        # Defensive fixes
        if temporal_features is None or np.asarray(temporal_features).ndim == 0:
            logger.error("Temporal features are empty. Please verify dataset columns.")
            return
        temporal_features = np.asarray(temporal_features)
        if temporal_features.ndim == 1:
            temporal_features = temporal_features.reshape(-1, 1)
    
    # Reshape for model input (same as in train)
    sequence_length = config.get('preprocessing.sequence_length')
    if temporal_features.shape[1] == 0:
        logger.error("No feature columns after feature extraction. Aborting.")
        return
    num_features = temporal_features.shape[1] // sequence_length if temporal_features.shape[1] >= sequence_length else temporal_features.shape[1]
    X_model = temporal_features.reshape(-1, sequence_length, num_features)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Setup explainers
    shap_explainer = SHAPExplainer(
        model,
        model_type='kernel',
        sequence_length=sequence_length,
        num_features=num_features
    )  # Use kernel for TF 2.x compatibility
    
    # Generate explanations
    background_data = X_model[:config.get('explainability.shap_background_samples')]
    shap_explainer.setup_explainer(background_data)
    
    # Explain a sample of instances
    sample_indices = np.random.choice(len(X_model), 10, replace=False)
    sample_data = X_model[sample_indices]
    
    # For sequence models, flatten the temporal dimension for SHAP
    # SHAP expects 2D input, so we flatten sequence_length × num_features
    flattened_sample_data = sample_data.reshape(sample_data.shape[0], -1)
    flattened_background = background_data.reshape(background_data.shape[0], -1)

    # Update the explainer with flattened data
    shap_explainer.setup_explainer(flattened_background)

    shap_values = shap_explainer.explain_dataset(flattened_sample_data)
    
    # Plot explanations (use flattened data for feature names)
    os.makedirs('results/plots', exist_ok=True)
    fig = shap_explainer.plot_summary(shap_values, flattened_sample_data)
    fig.savefig('results/plots/shap_summary.png')
    
    logger.info("Explanations generated and saved to results/plots/")

def main():
    args = parse_args()
    
    # Setup config
    config = Config(args.config)

    # Resolve data path: CLI arg takes precedence, otherwise use config value
    effective_data_path = args.data_path or config.get('data.raw_data_path')
    
    # Setup logging
    os.makedirs('results/logs', exist_ok=True)
    logger = setup_logger('vanet_xai', 'results/logs/vanet_xai.log')

    # Setup GPU (if available)
    setup_gpu(logger)
    
    # Apply CLI overrides to config if provided
    if args.epochs is not None:
        config.set('training.epochs', args.epochs)
    if args.batch_size is not None:
        config.set('training.batch_size', args.batch_size)
    if args.rows_limit is not None:
        # Keep this in config as well so it is visible to any downstream callers
        config.set('data.rows_limit', args.rows_limit)
    
    if args.mode == 'train':
        train(
            config,
            effective_data_path,
            logger,
            rows_limit=config.get('data.rows_limit'),
            cache_npz=args.cache_npz,
        )
    elif args.mode == 'evaluate':
        if args.model_path is None:
            logger.error("Model path must be provided for evaluate mode")
            return
        evaluate(
            config,
            args.model_path,
            effective_data_path,
            logger,
            rows_limit=config.get('data.rows_limit'),
            cache_npz=args.cache_npz,
        )
    elif args.mode == 'explain':
        if args.model_path is None:
            logger.error("Model path must be provided for explain mode")
            return
        explain(
            config,
            args.model_path,
            effective_data_path,
            logger,
            rows_limit=config.get('data.rows_limit'),
            cache_npz=args.cache_npz,
        )

if __name__ == "__main__":
    main()
