"""
Visualize temporal features and Conv1D model behavior.

Usage (features only, no model - avoids TensorFlow/PyTorch):
python visualize_temporal.py --features openmic-2018.npz --sample-idx 0

Usage (with PyTorch model):
python visualize_temporal.py --features openmic-2018.npz --model temporal_conv1d_openmic.pt --sample-idx 0

Usage (with Keras model):
python visualize_temporal.py --features openmic-2018.npz --model temporal_conv1d_openmic.h5 --sample-idx 0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Lazy-load TensorFlow (import at module load can cause segfault on some systems).
# Only imported when --model path to .h5 file is provided.
KERAS_AVAILABLE = None


def _ensure_keras():
    """Import Keras/TensorFlow only when needed."""
    global KERAS_AVAILABLE
    if KERAS_AVAILABLE is not None:
        return KERAS_AVAILABLE
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        KERAS_AVAILABLE = True
        return True
    except ImportError:
        KERAS_AVAILABLE = False
        return False


def load_features(npz_path):
    """Load features from npz file."""
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    
    prefer = ['arr_0', 'X', 'features', 'vggish', 'embeddings']
    feature_key = None
    for p in prefer:
        if p in keys:
            feature_key = p
            break
    if not feature_key:
        feature_key = keys[0]
    
    X = data[feature_key]
    
    # Get clip IDs if available
    clip_ids = None
    for candidate in ['clip_ids', 'clip_id', 'ids', 'filenames', 'file_names', 'sample_key']:
        if candidate in keys:
            clip_ids = np.asarray(data[candidate]).astype(str)
            break
    
    return X, clip_ids


def plot_temporal_features(features, sample_idx, clip_id=None, save_path='temporal_features.png'):
    """
    Visualize temporal features for a single sample.
    
    Args:
        features: 3D array (clips, frames, features)
        sample_idx: Index of sample to visualize
        clip_id: Optional clip identifier
        save_path: Where to save the plot
    """
    sample = features[sample_idx]  # Shape: (frames, features)
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # 1. Heatmap of all features over time
    ax1 = plt.subplot(gs[0, :])
    im = ax1.imshow(sample.T, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xlabel('Time Frame', fontsize=12)
    ax1.set_ylabel('Feature Dimension', fontsize=12)
    title = f'Temporal Feature Heatmap - Sample {sample_idx}'
    if clip_id:
        title += f' ({clip_id})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Feature Value')
    
    # 2. Average feature activation over time
    ax2 = plt.subplot(gs[1, 0])
    mean_activation = np.mean(sample, axis=1)
    ax2.plot(mean_activation, linewidth=2, color='#2E86AB')
    ax2.fill_between(range(len(mean_activation)), mean_activation, alpha=0.3, color='#2E86AB')
    ax2.set_xlabel('Time Frame', fontsize=11)
    ax2.set_ylabel('Mean Activation', fontsize=11)
    ax2.set_title('Average Feature Activation Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature variance over time
    ax3 = plt.subplot(gs[1, 1])
    variance = np.var(sample, axis=1)
    ax3.plot(variance, linewidth=2, color='#A23B72')
    ax3.fill_between(range(len(variance)), variance, alpha=0.3, color='#A23B72')
    ax3.set_xlabel('Time Frame', fontsize=11)
    ax3.set_ylabel('Variance', fontsize=11)
    ax3.set_title('Feature Variance Over Time', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved temporal features visualization to: {save_path}")
    plt.close()


def plot_conv1d_activations_keras(model, sample, save_path='conv1d_activations.png'):
    """
    Visualize Conv1D layer activations for a Keras model.
    
    Args:
        model: Trained Keras model
        sample: Single sample (frames, features)
        save_path: Where to save the plot
    """
    if not _ensure_keras():
        print("TensorFlow not available. Cannot visualize Keras model activations.")
        return
    from tensorflow import keras

    # Get outputs from each Conv1D layer
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if 'conv1d' in layer.name.lower():
            conv_layers.append((i, layer.name, layer))
    
    if not conv_layers:
        print("No Conv1D layers found in model.")
        return
    
    # Create intermediate models
    intermediate_outputs = []
    for idx, name, layer in conv_layers:
        intermediate_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[idx].output
        )
        activation = intermediate_model.predict(sample[np.newaxis, ...], verbose=0)
        intermediate_outputs.append((name, activation[0]))
    
    # Plot
    n_layers = len(intermediate_outputs)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers))
    
    if n_layers == 1:
        axes = [axes]
    
    for ax, (name, activation) in zip(axes, intermediate_outputs):
        # activation shape: (time_steps, filters)
        im = ax.imshow(activation.T, aspect='auto', origin='lower', cmap='coolwarm')
        ax.set_xlabel('Time Step (after pooling)', fontsize=11)
        ax.set_ylabel('Filter Index', fontsize=11)
        ax.set_title(f'Layer: {name} - Activation Map ({activation.shape[1]} filters)', 
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Activation')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Conv1D activations to: {save_path}")
    plt.close()


def plot_conv1d_activations(model, sample, save_path='conv1d_activations.png', backend='keras'):
    """Dispatch to Keras or PyTorch activation visualization."""
    if backend == 'pytorch':
        print("PyTorch model activation visualization: rebuild model from training script.")
        print("Use --model with .h5 (Keras) for activation viz, or omit --model for feature-only plots.")
        return
    plot_conv1d_activations_keras(model, sample, save_path)


def compare_temporal_vs_averaged(features, sample_idx, save_path='temporal_vs_averaged.png'):
    """
    Compare temporal features vs. time-averaged features.
    
    Args:
        features: 3D array (clips, frames, features)
        sample_idx: Index of sample to compare
        save_path: Where to save the plot
    """
    sample = features[sample_idx]  # (frames, features)
    averaged = np.mean(sample, axis=0)  # (features,)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temporal
    im1 = axes[0].imshow(sample.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_xlabel('Time Frame', fontsize=12)
    axes[0].set_ylabel('Feature Dimension', fontsize=12)
    axes[0].set_title('Temporal Features (Used by Conv1D)', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Value')
    
    # Averaged
    axes[1].plot(averaged, linewidth=2, color='#E63946')
    axes[1].fill_between(range(len(averaged)), averaged, alpha=0.3, color='#E63946')
    axes[1].set_xlabel('Feature Dimension', fontsize=12)
    axes[1].set_ylabel('Average Value', fontsize=12)
    axes[1].set_title('Time-Averaged Features (Used by RF)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved temporal vs averaged comparison to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize temporal features and Conv1D activations')
    parser.add_argument('--features', required=True, help='Path to .npz features file')
    parser.add_argument('--model', nargs='?', default=None, const=None,
                       help='Path to trained Conv1D model (.h5 or .pt). Omit or use --model with no value for feature-only viz (avoids TensorFlow)')
    parser.add_argument('--sample-idx', '--sample-id', dest='sample_idx', type=int, default=0,
                       help='Index of sample to visualize (default: 0)')
    parser.add_argument('--output-dir', default='.', 
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Validate: empty --model is treated as no model (avoids loading TensorFlow)
    if args.model is not None and (not isinstance(args.model, str) or not args.model.strip()):
        args.model = None
    
    # Load features
    print("Loading features...")
    features, clip_ids = load_features(args.features)
    
    if features.ndim == 2:
        print("Error: Features are 2D (already averaged). Need 3D temporal features.")
        print("Make sure your .npz file contains temporal (frame-level) features.")
        return
    
    print(f"Features shape: {features.shape} (clips, frames, features)")
    
    if args.sample_idx >= features.shape[0]:
        print(f"Error: sample_idx {args.sample_idx} out of range (max: {features.shape[0]-1})")
        return
    
    clip_id = clip_ids[args.sample_idx] if clip_ids is not None else None
    
    # Plot temporal features
    print(f"\nVisualizing sample {args.sample_idx}...")
    plot_temporal_features(
        features, 
        args.sample_idx, 
        clip_id,
        save_path=f'{args.output_dir}/temporal_features_sample_{args.sample_idx}.png'
    )
    
    # Compare temporal vs averaged
    compare_temporal_vs_averaged(
        features,
        args.sample_idx,
        save_path=f'{args.output_dir}/temporal_vs_averaged_sample_{args.sample_idx}.png'
    )
    
    # Plot Conv1D activations if model provided (lazy-load TensorFlow only here)
    if args.model and args.model.strip():
        model_path = args.model.strip()
        if model_path.lower().endswith('.pt'):
            print("\nPyTorch (.pt) model: activation viz requires Keras .h5. Skipping model viz.")
            print("Feature-only visualizations saved above.")
        elif model_path.lower().endswith('.h5'):
            if not _ensure_keras():
                print("\nWarning: TensorFlow not available. Cannot load Keras model.")
            else:
                from tensorflow import keras
                print(f"\nLoading Keras model from {model_path}...")
                model = keras.models.load_model(model_path)
                print("Model loaded successfully.")
                sample = features[args.sample_idx]
                plot_conv1d_activations(
                    model, sample,
                    save_path=f'{args.output_dir}/conv1d_activations_sample_{args.sample_idx}.png',
                    backend='keras'
                )
        else:
            print(f"\nUnknown model format: {model_path}. Use .h5 (Keras) or .pt (PyTorch).")
    
    print("\nVisualization complete!")
    print(f"Check the {args.output_dir} directory for output images.")


if __name__ == '__main__':
    main()
