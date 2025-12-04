#!/usr/bin/env python3
"""
Main training script for music genre classification.
"""

import os
import sys
import argparse
import yaml
import json
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DeepBottleneckAE, EnhancedGenreClassifier
from src.dataset import MelSpectrogramDataset
from src.train import train_autoencoder, train_classifier
from src.utils import set_seed, plot_training_history
from torch.utils.data import DataLoader, random_split


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description="Train music genre classification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data_segments.json",
        help="Path to data segments JSON file"
    )
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip autoencoder training (use existing checkpoint)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    dataset = MelSpectrogramDataset(args.data)
    genres = dataset.genres
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Phase 0: Train Autoencoder
    if not args.skip_ae:
        print("\n" + "="*70)
        print("PHASE 0: AUTOENCODER PRE-TRAINING")
        print("="*70)
        
        ae_model = DeepBottleneckAE(latent_dim=config['model']['latent_dim'])
        
        ae_history = train_autoencoder(
            model=ae_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['ae']['epochs'],
            learning_rate=config['training']['ae']['learning_rate'],
            device=device,
            checkpoint_dir=config['paths']['checkpoints_dir'],
            patience=config['training']['ae']['patience'],
            seed=config['seed']
        )
        
        # Plot and save training history
        plot_training_history(
            ae_history,
            save_path=os.path.join(config['paths']['plots_dir'], 'ae_training_history.png')
        )
    else:
        print("Skipping autoencoder training (using existing checkpoint)")
    
    # Phase 1: Train Classifier with Frozen Encoder
    print("\n" + "="*70)
    print("PHASE 1: CLASSIFIER TRAINING (FROZEN ENCODER)")
    print("="*70)
    
    # Load pre-trained autoencoder
    ae_model = DeepBottleneckAE(latent_dim=config['model']['latent_dim'])
    ae_checkpoint = os.path.join(config['paths']['checkpoints_dir'], 'best_autoencoder.pth')
    ae_model.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    
    # Create classifier with pre-trained encoder
    classifier = EnhancedGenreClassifier(
        encoder=ae_model.encoder,
        latent_dim=config['model']['latent_dim'],
        num_genres=config['model']['num_genres']
    )
    
    phase1_history = train_classifier(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['phase1']['epochs'],
        learning_rate=config['training']['phase1']['learning_rate'],
        device=device,
        checkpoint_dir=config['paths']['checkpoints_dir'],
        freeze_encoder=config['training']['phase1']['freeze_encoder'],
        patience=config['training']['phase1']['patience'],
        seed=config['seed']
    )
    
    # Phase 2: Fine-tune Classifier
    print("\n" + "="*70)
    print("PHASE 2: CLASSIFIER FINE-TUNING (UNFROZEN ENCODER)")
    print("="*70)
    
    # Load best model from phase 1
    phase1_checkpoint = os.path.join(config['paths']['checkpoints_dir'], 'best_classifier.pth')
    classifier.load_state_dict(torch.load(phase1_checkpoint, map_location=device))
    
    # Unfreeze encoder for fine-tuning
    for param in classifier.encoder.parameters():
        param.requires_grad = True
    
    phase2_history = train_classifier(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['phase2']['epochs'],
        learning_rate=config['training']['phase2']['learning_rate'],
        device=device,
        checkpoint_dir=config['paths']['checkpoints_dir'],
        freeze_encoder=config['training']['phase2']['freeze_encoder'],
        patience=config['training']['phase2']['patience'],
        seed=config['seed']
    )
    
    # Save combined training history
    combined_history = {
        'phase1': phase1_history,
        'phase2': phase2_history
    }
    
    # Plot training history
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(phase1_history['train_acc'], label='Train', linewidth=2)
    axes[0, 0].plot(phase1_history['val_acc'], label='Val', linewidth=2)
    axes[0, 0].set_title('Phase 1: Accuracy (Frozen Encoder)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(phase1_history['train_loss'], label='Train', linewidth=2)
    axes[0, 1].plot(phase1_history['val_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('Phase 1: Loss (Frozen Encoder)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(phase2_history['train_acc'], label='Train', linewidth=2)
    axes[1, 0].plot(phase2_history['val_acc'], label='Val', linewidth=2)
    axes[1, 0].set_title('Phase 2: Accuracy (Fine-Tuning)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(phase2_history['train_loss'], label='Train', linewidth=2)
    axes[1, 1].plot(phase2_history['val_loss'], label='Val', linewidth=2)
    axes[1, 1].set_title('Phase 2: Loss (Fine-Tuning)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(config['paths']['plots_dir'], 'training_history_combined.png'),
        dpi=150, bbox_inches='tight'
    )
    print(f"\nSaved training history to {config['paths']['plots_dir']}/training_history_combined.png")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

