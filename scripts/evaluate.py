#!/usr/bin/env python3
"""
Evaluation script for music genre classification model.
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DeepBottleneckAE, EnhancedGenreClassifier
from src.dataset import MelSpectrogramDataset
from src.train import evaluate
from src.utils import plot_confusion_matrix
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
    parser = argparse.ArgumentParser(description="Evaluate music genre classification model")
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
        "--checkpoint",
        type=str,
        default="results/checkpoints/best_classifier.pth",
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    dataset = MelSpectrogramDataset(args.data)
    genres = dataset.genres
    
    # Split data (same split as training)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    
    # Load autoencoder to get encoder
    ae_model = DeepBottleneckAE(latent_dim=config['model']['latent_dim'])
    ae_checkpoint = os.path.join(config['paths']['checkpoints_dir'], 'best_autoencoder.pth')
    ae_model.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    
    # Create classifier
    classifier = EnhancedGenreClassifier(
        encoder=ae_model.encoder,
        latent_dim=config['model']['latent_dim'],
        num_genres=config['model']['num_genres']
    )
    
    # Load classifier weights
    classifier.load_state_dict(torch.load(args.checkpoint, map_location=device))
    classifier = classifier.to(device)
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_acc, conf_matrix, class_report = evaluate(
        model=classifier,
        test_loader=test_loader,
        device=device,
        genres=genres
    )
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nPer-class metrics:")
    for genre in genres:
        if genre in class_report:
            metrics = class_report[genre]
            print(f"  {genre:12s} - Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    
    # Save results
    os.makedirs(config['paths']['metrics_dir'], exist_ok=True)
    
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    results_path = os.path.join(config['paths']['metrics_dir'], 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        conf_matrix=conf_matrix,
        genres=genres,
        title="Confusion Matrix: Music Genre Classification",
        save_path=os.path.join(config['paths']['plots_dir'], 'confusion_matrix.png')
    )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

