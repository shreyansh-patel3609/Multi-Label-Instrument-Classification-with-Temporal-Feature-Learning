# openmic_rf_from_longcsv.py
"""
Usage example (run in VS Code terminal):

python rf_baseline_openmic.py \
  --features openmic-2018.npz \
  --labels openmic-2018-aggregated-labels.csv \
  --out rf_openmic.joblib \
  --relevance-threshold 0.5

What it does (simple):
- Converts long CSV (sample_key, instrument, relevance, num_responses) into one row per clip.
- Binarizes relevance >= threshold to 1, else 0.
- Trains a One-vs-Rest RandomForest on VGGish features.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import sys
import os

def load_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    if not keys:
        raise ValueError(f"No arrays found in {npz_path}")
    # pick a reasonable feature array
    prefer = ['arr_0', 'X', 'features', 'vggish', 'embeddings']
    feature_key = None
    for p in prefer:
        if p in keys:
            feature_key = p
            break
    if feature_key is None:
        # choose the largest 2D/3D array
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
    # if features are per-frame (3D), reduce to per-clip by averaging over frames
    if X.ndim == 3:
        X = np.mean(X, axis=1)
    meta = {}
    # look for clip ids in other arrays
    for candidate in ['clip_ids', 'clip_id', 'ids', 'filenames', 'file_names', 'sample_key']:
        if candidate in keys:
            meta['clip_id'] = np.asarray(data[candidate]).astype(str)
            break
    # also try to detect a mapping stored as structured array
    return X, meta

def pivot_long_csv_to_labels(df_long, relevance_col='relevance', sample_col='sample_key', instrument_col='instrument',
                             threshold=0.5, aggfunc='max', save_csv=None):
    """
    Convert long-format csv (one row per sample_key+instrument) to wide binary label matrix.
    - aggfunc: how to combine multiple relevance values for same sample/instrument (default 'max')
    - threshold: relevance >= threshold => label 1, else 0
    Returns: labels_df (index=sample_key, columns=instrument, values 0/1)
    """
    # Keep only necessary columns (guard against other extra columns)
    df = df_long[[sample_col, instrument_col, relevance_col]].copy()
    # sometimes instrument names have whitespace; strip them
    df[instrument_col] = df[instrument_col].astype(str).str.strip()
    # pivot to table of relevance scores
    pivot = df.pivot_table(index=sample_col, columns=instrument_col, values=relevance_col, aggfunc=aggfunc)
    # fill missing relevance with 0.0 (no annotator indicated presence)
    pivot = pivot.fillna(0.0)
    # Binarize with threshold
    labels_bin = (pivot >= threshold).astype(int)
    if save_csv:
        labels_bin.to_csv(save_csv)
    return labels_bin

def safe_roc_auc(y_true, y_score, labels):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='Path to .npz features file')
    parser.add_argument('--labels', required=True, help='Path to long-format CSV (sample_key,instrument,relevance,...)')
    parser.add_argument('--relevance-threshold', type=float, default=0.5,
                        help='Threshold on relevance to binarize labels (default 0.5)')
    parser.add_argument('--out', default='rf_openmic.joblib', help='Path to save trained model')
    parser.add_argument('--save-labels-csv', default='labels_wide.csv', help='Save pivoted binarized labels here')
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    print("Loading features from:", args.features)
    X, meta = load_features(args.features)
    print("Features shape:", X.shape)

    print("Loading long CSV labels from:", args.labels)
    df_long = pd.read_csv(args.labels)
    # check expected columns quickly
    expected = {'sample_key', 'instrument', 'relevance'}
    if not expected.issubset(set(df_long.columns)):
        print("Warning: CSV does not contain the exact columns 'sample_key','instrument','relevance'.")
        print("Found columns:", list(df_long.columns))
        print("If names differ, you can rename columns or adjust the script arguments.")
        # Try to continue: try case-insensitive mapping
        cols_lower = {c.lower(): c for c in df_long.columns}
        col_map = {}
        for want in expected:
            if want in df_long.columns:
                col_map[want] = want
            elif want in cols_lower:
                col_map[want] = cols_lower[want]
        if len(col_map) < 3:
            raise RuntimeError("Required columns not found. CSV columns must include sample_key, instrument, relevance (case-insensitive).")
        # rename
        df_long = df_long.rename(columns={v: k for k, v in col_map.items()})

    # Pivot and binarize
    print("Pivoting long CSV into wide label matrix and binarizing with threshold:", args.relevance_threshold)
    labels_df = pivot_long_csv_to_labels(df_long, threshold=args.relevance_threshold, save_csv=args.save_labels_csv)
    print("Pivoted labels shape (clips x instruments):", labels_df.shape)
    print(f"Binarized labels saved to: {args.save_labels_csv}")

    # Prepare label matrix and ensure clip id alignment
    y_index = labels_df.index.astype(str).values  # sample_key strings
    y = labels_df.values  # shape (n_clips_labels, n_instruments)
    label_cols = list(labels_df.columns)

    # If features file contains clip ids (meta['clip_id']), align by that; otherwise attempt to match order
    if 'clip_id' in meta:
        feat_ids = np.asarray(meta['clip_id']).astype(str)
        print("Detected clip ids inside features file. Aligning labels to features using sample_key/clip_id.")
        # Build map from label index (sample_key -> row index in labels_df)
        idx_map = {k: i for i, k in enumerate(y_index)}
        sel = [idx_map.get(fid, None) for fid in feat_ids]
        if any(s is None for s in sel):
            missing = [fid for fid, s in zip(feat_ids, sel) if s is None]
            print("Warning: Some feature clip ids were not found in labels sample_key list.")
            print("Missing example IDs (first 10):", missing[:10])
            # We'll proceed by selecting only those that match
            matched_pairs = [(i, idx_map[fid]) for i, fid in enumerate(feat_ids) if fid in idx_map]
            if len(matched_pairs) == 0:
                raise RuntimeError("No overlap between features clip ids and CSV sample_key values. Can't align.")
            feat_indices, label_indices = zip(*matched_pairs)
            X = X[list(feat_indices)]
            y = y[list(label_indices)]
            print(f"Aligned using {len(feat_indices)} matching clips (dropped unmatched). New feature shape: {X.shape}, labels: {y.shape}")
        else:
            # exact alignment
            sel = np.array(sel, dtype=int)
            y = y[sel]
            print("Exact alignment succeeded. Feature rows match label rows.")
    else:
        # no clip ids in features: only proceed if number of rows match, otherwise error
        if X.shape[0] != y.shape[0]:
            raise RuntimeError(f"Feature rows ({X.shape[0]}) do not match number of label rows ({y.shape[0]}) and no clip ids present to align by.")
        else:
            print("No clip ids in features, but row counts match; assuming same order.")

    print("Final X shape:", X.shape, "Final y shape:", y.shape)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    # Train classifier
    print("Training One-vs-Rest RandomForest (may take a while)...")
    base = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=args.random_state)
    clf = OneVsRestClassifier(base)
    clf.fit(X_train, y_train)

    # Predict & score
    print("Predicting on test set...")
    y_pred = clf.predict(X_test)
    try:
        y_score = clf.predict_proba(X_test)
        # Create a DataFrame for predictions
        pred_df = pd.DataFrame(
        y_pred,
        columns=label_cols
        )   
    except Exception:
        try:
            y_score = clf.decision_function(X_test)
        except Exception:
            y_score = y_pred.astype(float)

    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_avg, per_class_auc = safe_roc_auc(y_test, np.array(y_score), label_cols)

    print("\n=== RESULTS ===")
    print("Micro F1 :", round(micro_f1, 4))
    print("Macro F1 :", round(macro_f1, 4))
    if auc_avg is None:
        print("Macro AUC: None (no class had both positive and negative samples in test set)")
    else:
        print("Macro AUC :", round(auc_avg, 4))

    print("\nPer-class AUC (None means could not compute):")
    for k, v in per_class_auc.items():
        print(f"  {k}: {round(v, 4) if v is not None else 'None'}")

    # Save model
    joblib.dump(clf, args.out)
    print(f"Saved trained model to: {args.out}")
    # Save to CSV
    pred_df.to_csv("predicted_labels.csv", index=False)
    print("Saved predicted binary labels to predicted_labels.csv")

    #Also save probability scores (if available)
    try:
        score_df = pd.DataFrame(
        y_score,
        columns=label_cols
        )
        score_df.to_csv("predicted_scores.csv", index=False)
        print("Saved predicted probability scores to predicted_scores.csv")
    except:
        print("Could not save probability scores (model didn't produce them).")

if __name__ == '__main__':
    main()		
