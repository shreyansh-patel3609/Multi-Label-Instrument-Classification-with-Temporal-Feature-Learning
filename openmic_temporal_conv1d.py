"""
Temporal extension with Conv1D for OpenMIC baseline.

Usage example:

python openmic_temporal_conv1d.py \
  --features openmic-2018.npz \
  --labels openmic-2018-aggregated-labels.csv \
  --out temporal_conv1d_openmic.h5 \
  --relevance-threshold 0.5 \
  --model-type rf

For Conv1D (requires TensorFlow; may cause segfault on some systems):

python openmic_temporal_conv1d.py \
  --features openmic-2018.npz \
  --labels openmic-2018-aggregated-labels.csv \
  --out temporal_conv1d_openmic.h5 \
  --model-type conv1d

What it does:
- Preserves temporal structure (3D: clips × frames × features)
- Uses 1D CNN across time dimension to capture temporal patterns
- Supports both the original RF baseline and new temporal Conv1D model
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import os

# Lazy-load deep learning backends (avoid TensorFlow segfault on some systems).
# PyTorch is tried first for Conv1D; TensorFlow/Keras used only if --backend keras.
PYTORCH_AVAILABLE = None
_torch = None
_nn = None
KERAS_AVAILABLE = None
_tf = None
_keras = None
_layers = None
_models = None


def _ensure_pytorch():
    """Import PyTorch only when needed. Preferred for Conv1D (avoids TensorFlow segfault)."""
    global PYTORCH_AVAILABLE, _torch, _nn
    if PYTORCH_AVAILABLE is not None:
        return PYTORCH_AVAILABLE
    try:
        import torch
        import torch.nn as nn
        _torch, _nn = torch, nn
        PYTORCH_AVAILABLE = True
        return True
    except ImportError:
        PYTORCH_AVAILABLE = False
        return False


def _ensure_keras():
    """Import TensorFlow/Keras only when needed. Avoids segfault at startup for RF-only usage."""
    global KERAS_AVAILABLE, _tf, _keras, _layers, _models
    if KERAS_AVAILABLE is not None:
        return KERAS_AVAILABLE
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models
        _tf, _keras, _layers, _models = tf, keras, layers, models
        KERAS_AVAILABLE = True
        return True
    except ImportError:
        KERAS_AVAILABLE = False
        print("Warning: TensorFlow/Keras not available. Only RandomForest model will work.")
        return False


def load_features(npz_path, preserve_temporal=False):
    """
    Load features with option to preserve temporal dimension.
    
    Args:
        npz_path: Path to .npz file
        preserve_temporal: If True, keep 3D shape (clips, frames, features)
                          If False, average over frames to get 2D (clips, features)
    
    Returns:
        X: Feature array
        meta: Dictionary with metadata (e.g., clip_ids)
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    if not keys:
        raise ValueError(f"No arrays found in {npz_path}")
    
    # Find feature array
    prefer = ['arr_0', 'X', 'features', 'vggish', 'embeddings']
    feature_key = None
    for p in prefer:
        if p in keys:
            feature_key = p
            break
    if feature_key is None:
        best_k = None
        best_size = -1
        for k in keys:
            a = data[k]
            if hasattr(a, 'ndim') and a.ndim >= 2:
                size = a.shape[0] * (a.shape[1] if a.ndim > 1 else 1)
                if size > best_size:
                    best_size = size
                    best_k = k
        feature_key = best_k or keys[0]

    X = data[feature_key]
    
    # Handle temporal dimension
    if X.ndim == 3:
        if preserve_temporal:
            print(f"Preserving temporal structure: {X.shape} (clips × frames × features)")
        else:
            print(f"Averaging over temporal dimension: {X.shape} -> {(X.shape[0], X.shape[2])}")
            X = np.mean(X, axis=1)
    elif X.ndim == 2:
        if preserve_temporal:
            print(f"Warning: Features are 2D {X.shape}, cannot preserve temporal structure.")
            print("Adding dummy temporal dimension of size 1.")
            X = np.expand_dims(X, axis=1)
    
    # Load metadata
    meta = {}
    for candidate in ['clip_ids', 'clip_id', 'ids', 'filenames', 'file_names', 'sample_key']:
        if candidate in keys:
            meta['clip_id'] = np.asarray(data[candidate]).astype(str)
            break
    
    return X, meta


def pivot_long_csv_to_labels(df_long, relevance_col='relevance', sample_col='sample_key', 
                             instrument_col='instrument', threshold=0.5, aggfunc='max', save_csv=None):
    """Convert long-format csv to wide binary label matrix."""
    df = df_long[[sample_col, instrument_col, relevance_col]].copy()
    df[instrument_col] = df[instrument_col].astype(str).str.strip()
    pivot = df.pivot_table(index=sample_col, columns=instrument_col, values=relevance_col, aggfunc=aggfunc)
    pivot = pivot.fillna(0.0)
    labels_bin = (pivot >= threshold).astype(int)
    if save_csv:
        labels_bin.to_csv(save_csv)
    return labels_bin


def build_temporal_conv1d_model(input_shape, num_classes, config=None):
    """
    Build a 1D CNN model for temporal sequence classification.
    
    Args:
        input_shape: (time_steps, features) - shape of one sample
        num_classes: Number of output classes
        config: Optional dict with model hyperparameters
    
    Returns:
        Compiled Keras model
    """
    if not _ensure_keras():
        raise ImportError("TensorFlow/Keras required for temporal Conv1D model")
    
    # Default configuration
    if config is None:
        config = {}
    
    filters = config.get('filters', [64, 128, 256])
    kernel_size = config.get('kernel_size', 3)
    pool_size = config.get('pool_size', 2)
    dense_units = config.get('dense_units', 256)
    dropout_rate = config.get('dropout_rate', 0.5)
    
    model = _models.Sequential(name='TemporalConv1D')
    
    # First Conv1D block
    model.add(_layers.Conv1D(filters[0], kernel_size, activation='relu', 
                            padding='same', input_shape=input_shape))
    model.add(_layers.BatchNormalization())
    model.add(_layers.MaxPooling1D(pool_size))
    model.add(_layers.Dropout(dropout_rate * 0.5))
    
    # Additional Conv1D blocks
    for f in filters[1:]:
        model.add(_layers.Conv1D(f, kernel_size, activation='relu', padding='same'))
        model.add(_layers.BatchNormalization())
        model.add(_layers.MaxPooling1D(pool_size))
        model.add(_layers.Dropout(dropout_rate * 0.5))
    
    # Global pooling to aggregate temporal information
    model.add(_layers.GlobalAveragePooling1D())
    
    # Dense layers
    model.add(_layers.Dense(dense_units, activation='relu'))
    model.add(_layers.Dropout(dropout_rate))
    
    # Output layer (sigmoid for multi-label classification)
    model.add(_layers.Dense(num_classes, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=_keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            _tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return model


def _build_temporal_conv1d_pytorch(input_shape, num_classes, config=None):
    """Build PyTorch Conv1D model. Same architecture as Keras version."""
    if not _ensure_pytorch():
        raise ImportError("PyTorch required for Conv1D model. Install: pip install torch")
    nn = _nn
    torch = _torch

    class TemporalConv1DPyTorch(nn.Module):
        """1D CNN for temporal sequence classification."""

        def __init__(self, input_channels, num_classes, filters, kernel_size, pool_size, dense_units, dropout_rate):
            super().__init__()
            layers_list = []
            in_ch = input_channels
            for f in filters:
                layers_list.append(nn.Conv1d(in_ch, f, kernel_size, padding=kernel_size // 2))
                layers_list.append(nn.BatchNorm1d(f))
                layers_list.append(nn.ReLU())
                layers_list.append(nn.MaxPool1d(pool_size))
                layers_list.append(nn.Dropout(dropout_rate * 0.5))
                in_ch = f

            self.conv_blocks = nn.Sequential(*layers_list)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(filters[-1], dense_units)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(dense_units, num_classes)

        def forward(self, x):
            x = x.transpose(1, 2)  # (batch, features, time_steps)
            x = self.conv_blocks(x)
            x = self.global_pool(x)
            x = x.squeeze(-1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.fc2(x))
            return x

    if config is None:
        config = {}
    time_steps, input_channels = input_shape[0], input_shape[1]
    return TemporalConv1DPyTorch(
        input_channels=input_channels,
        num_classes=num_classes,
        filters=tuple(config.get('filters', [64, 128, 256])),
        kernel_size=config.get('kernel_size', 3),
        pool_size=config.get('pool_size', 2),
        dense_units=config.get('dense_units', 256),
        dropout_rate=config.get('dropout_rate', 0.5),
    )


def safe_roc_auc(y_true, y_score, labels):
    """Calculate ROC AUC with handling for classes with single value."""
    aucs = []
    per_class = {}
    for i, lab in enumerate(labels):
        col_true = y_true[:, i]
        if len(np.unique(col_true)) < 2:
            per_class[lab] = None
            continue
        try:
            a = roc_auc_score(col_true, y_score[:, i])
            per_class[lab] = float(a)
            aucs.append(a)
        except Exception:
            per_class[lab] = None
    avg = float(np.mean(aucs)) if aucs else None
    return avg, per_class


def train_random_forest(X_train, y_train, X_test, y_test, label_cols, args):
    """Train and evaluate RandomForest baseline."""
    print("\n=== Training RandomForest Baseline ===")
    print("Training One-vs-Rest RandomForest...")
    
    base = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, 
                                  random_state=args.random_state)
    clf = OneVsRestClassifier(base)
    clf.fit(X_train, y_train)
    
    print("Predicting on test set...")
    y_pred = clf.predict(X_test)
    
    try:
        y_score = clf.predict_proba(X_test)
    except Exception:
        try:
            y_score = clf.decision_function(X_test)
        except Exception:
            y_score = y_pred.astype(float)
    
    return clf, y_pred, y_score


def train_temporal_conv1d_pytorch(X_train, y_train, X_test, y_test, label_cols, args):
    """Train and evaluate temporal Conv1D model (PyTorch backend)."""
    if not _ensure_pytorch():
        raise ImportError("PyTorch required for Conv1D model. Install: pip install torch")
    device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
    print(f"\n=== Training Temporal Conv1D Model (PyTorch) on {device} ===")

    config = {
        'filters': [64, 128, 256],
        'kernel_size': 3,
        'pool_size': 2,
        'dense_units': 256,
        'dropout_rate': 0.5,
    }
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    model = _build_temporal_conv1d_pytorch(input_shape, num_classes, config)
    model = model.to(device)

    X_train_t = _torch.FloatTensor(X_train)
    y_train_t = _torch.FloatTensor(y_train)
    X_test_t = _torch.FloatTensor(X_test)
    y_test_t = _torch.FloatTensor(y_test)

    criterion = _nn.BCELoss()
    optimizer = _torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = _torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        perm = _torch.randperm(X_train_t.size(0))
        for i in range(0, X_train_t.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            bx = X_train_t[idx].to(device)
            by = y_train_t[idx].to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with _torch.no_grad():
            val_out = model(X_test_t.to(device))
            val_loss = criterion(val_out, y_test_t.to(device)).item()

        scheduler.step(val_loss)
        history['loss'].append(epoch_loss / (X_train_t.size(0) // args.batch_size + 1))
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{args.epochs} - loss: {history['loss'][-1]:.4f} - val_loss: {val_loss:.4f}")

        if patience_counter >= args.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with _torch.no_grad():
        y_score = model(X_test_t.to(device)).cpu().numpy()
    y_pred = (y_score >= 0.5).astype(int)

    return model, y_pred, y_score, type('History', (), {'history': history})()


def train_temporal_conv1d(X_train, y_train, X_test, y_test, label_cols, args):
    """Train and evaluate temporal Conv1D model (Keras/TensorFlow backend)."""
    if not _ensure_keras():
        raise ImportError("TensorFlow/Keras required for Conv1D model")
    
    print("\n=== Training Temporal Conv1D Model ===")
    
    # Build model configuration
    config = {
        'filters': [64, 128, 256],
        'kernel_size': 3,
        'pool_size': 2,
        'dense_units': 256,
        'dropout_rate': 0.5,
        'learning_rate': args.learning_rate
    }
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    num_classes = y_train.shape[1]
    
    model = build_temporal_conv1d_model(input_shape, num_classes, config)
    print(model.summary())
    
    # Callbacks
    callbacks = [
        _keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            restore_best_weights=True
        ),
        _keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Predict
    print("Predicting on test set...")
    y_score = model.predict(X_test, verbose=0)
    y_pred = (y_score >= 0.5).astype(int)
    
    return model, y_pred, y_score, history


def evaluate_predictions(y_test, y_pred, y_score, label_cols):
    """Evaluate and print metrics."""
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_avg, per_class_auc = safe_roc_auc(y_test, np.array(y_score), label_cols)
    
    print("\n=== RESULTS ===")
    print(f"Micro F1 : {round(micro_f1, 4)}")
    print(f"Macro F1 : {round(macro_f1, 4)}")
    if auc_avg is None:
        print("Macro AUC: None (no class had both positive and negative samples in test set)")
    else:
        print(f"Macro AUC : {round(auc_avg, 4)}")
    
    print("\nPer-class AUC (None means could not compute):")
    for k, v in per_class_auc.items():
        val_str = f"{round(v, 4)}" if v is not None else "None"
        print(f"  {k}: {val_str}")
    
    return micro_f1, macro_f1, auc_avg


def main():
    parser = argparse.ArgumentParser(description='OpenMIC baseline with temporal Conv1D extension')
    
    # Data arguments
    parser.add_argument('--features', required=True, help='Path to .npz features file')
    parser.add_argument('--labels', required=True, help='Path to long-format CSV')
    parser.add_argument('--relevance-threshold', type=float, default=0.5,
                       help='Threshold on relevance to binarize labels')
    parser.add_argument('--save-labels-csv', default='labels_wide.csv',
                       help='Save pivoted binarized labels here')
    
    # Model selection
    parser.add_argument('--model-type', choices=['rf', 'conv1d'], default='conv1d',
                       help='Model type: rf=RandomForest, conv1d=Temporal 1D CNN (default)')
    parser.add_argument('--backend', choices=['pytorch', 'keras'], default='pytorch',
                       help='Backend for Conv1D: pytorch (default, avoids TensorFlow segfault) or keras')
    parser.add_argument('--out', default='model_openmic.h5',
                       help='Path to save model (.joblib for RF, .pt for PyTorch Conv1D, .h5 for Keras Conv1D)')
    
    # RandomForest parameters
    parser.add_argument('--n_estimators', type=int, default=200,
                       help='Number of trees for RandomForest')
    
    # Conv1D parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs for Conv1D')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for Conv1D training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for Conv1D')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    
    # General parameters
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set fraction')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Validate required paths
    if not args.features or not args.features.strip():
        parser.error("--features path cannot be empty")
    if not args.labels or not args.labels.strip():
        parser.error("--labels path cannot be empty. Example: --labels openmic-2018-aggregated-labels.csv")
    
    # Check model type compatibility (PyTorch preferred for Conv1D; Keras only if --backend keras)
    if args.model_type == 'conv1d':
        if args.backend == 'pytorch':
            if not _ensure_pytorch():
                print("Error: PyTorch not available. Install: pip install torch")
                print("Alternatively use --backend keras (requires TensorFlow)")
                sys.exit(1)
        else:
            if not _ensure_keras():
                print("Error: TensorFlow/Keras not available. Install: pip install tensorflow")
                print("Alternatively use --backend pytorch (default)")
                sys.exit(1)
    
    # Load features
    print("Loading features from:", args.features)
    preserve_temporal = (args.model_type == 'conv1d')
    X, meta = load_features(args.features, preserve_temporal=preserve_temporal)
    print("Features shape:", X.shape)
    
    # Load and pivot labels
    print("Loading long CSV labels from:", args.labels)
    df_long = pd.read_csv(args.labels)
    
    # Handle column name variations
    expected = {'sample_key', 'instrument', 'relevance'}
    if not expected.issubset(set(df_long.columns)):
        cols_lower = {c.lower(): c for c in df_long.columns}
        col_map = {}
        for want in expected:
            if want in df_long.columns:
                col_map[want] = want
            elif want in cols_lower:
                col_map[want] = cols_lower[want]
        if len(col_map) < 3:
            raise RuntimeError("Required columns not found. Need: sample_key, instrument, relevance")
        df_long = df_long.rename(columns={v: k for k, v in col_map.items()})
    
    print(f"Pivoting long CSV and binarizing with threshold: {args.relevance_threshold}")
    labels_df = pivot_long_csv_to_labels(df_long, threshold=args.relevance_threshold, 
                                        save_csv=args.save_labels_csv)
    print("Pivoted labels shape (clips x instruments):", labels_df.shape)
    print(f"Binarized labels saved to: {args.save_labels_csv}")
    
    # Align features and labels
    y_index = labels_df.index.astype(str).values
    y = labels_df.values
    label_cols = list(labels_df.columns)
    
    if 'clip_id' in meta:
        feat_ids = np.asarray(meta['clip_id']).astype(str)
        print("Aligning labels to features using clip IDs...")
        idx_map = {k: i for i, k in enumerate(y_index)}
        matched_pairs = [(i, idx_map[fid]) for i, fid in enumerate(feat_ids) if fid in idx_map]
        
        if len(matched_pairs) == 0:
            raise RuntimeError("No overlap between feature clip IDs and label sample_keys")
        
        feat_indices, label_indices = zip(*matched_pairs)
        X = X[list(feat_indices)]
        y = y[list(label_indices)]
        print(f"Aligned {len(feat_indices)} clips. New shapes - X: {X.shape}, y: {y.shape}")
    else:
        if X.shape[0] != y.shape[0]:
            raise RuntimeError(f"Feature rows ({X.shape[0]}) != label rows ({y.shape[0]})")
        print("No clip IDs found; assuming same order.")
    
    print(f"\nFinal X shape: {X.shape}, Final y shape: {y.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Train model based on type
    if args.model_type == 'rf':
        model, y_pred, y_score = train_random_forest(
            X_train, y_train, X_test, y_test, label_cols, args
        )
        # Save RF model
        joblib.dump(model, args.out)
        print(f"\nSaved RandomForest model to: {args.out}")
        
    elif args.model_type == 'conv1d':
        if args.backend == 'pytorch':
            model, y_pred, y_score, history = train_temporal_conv1d_pytorch(
                X_train, y_train, X_test, y_test, label_cols, args
            )
            out_path = args.out.replace('.h5', '.pt') if args.out.endswith('.h5') else args.out
            _torch.save({'model_state_dict': model.state_dict()}, out_path)
            print(f"\nSaved PyTorch Conv1D model to: {out_path}")
        else:
            model, y_pred, y_score, history = train_temporal_conv1d(
                X_train, y_train, X_test, y_test, label_cols, args
            )
            model.save(args.out)
            print(f"\nSaved Keras Conv1D model to: {args.out}")

        # Save training history
        history_csv = args.out.replace('.h5', '_history.csv').replace('.pt', '_history.csv')
        history_df = pd.DataFrame(history.history if hasattr(history, 'history') else history)
        history_df.to_csv(history_csv, index=False)
        print(f"Saved training history to: {history_csv}")
    
    # Evaluate
    evaluate_predictions(y_test, y_pred, y_score, label_cols)
    
    # Save predictions
    pred_df = pd.DataFrame(y_pred, columns=label_cols)
    pred_df.to_csv("predicted_labels.csv", index=False)
    print("\nSaved predicted binary labels to predicted_labels.csv")
    
    score_df = pd.DataFrame(y_score, columns=label_cols)
    score_df.to_csv("predicted_scores.csv", index=False)
    print("Saved predicted probability scores to predicted_scores.csv")


if __name__ == '__main__':
    main()
