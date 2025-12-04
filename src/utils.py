"""
Utility functions for data processing, visualization, and common operations.
"""

import os
import json
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_metadata_robust(dataset_path: str) -> pd.DataFrame:
    """
    Scans the GTZAN directory and extracts metadata from each .wav file.
    This version includes a try-except block to handle potential file loading errors.
    
    Args:
        dataset_path: Path to the genres_original directory
        
    Returns:
        DataFrame with metadata (filename, filepath, genre, duration, sample_rate)
    """
    metadata_list = []
    # Ensure genres are processed in a consistent order
    genres = sorted([g for g in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, g))])
    
    print("Building robust metadata file...")
    for genre in tqdm(genres, desc="Processing genres"):
        genre_path = os.path.join(dataset_path, genre)
        for filename in sorted(os.listdir(genre_path)):
            if filename.endswith('.wav'):
                filepath = os.path.join(genre_path, filename)
                try:
                    # Attempt to load audio file to verify it's not corrupt and get info
                    y, sr = librosa.load(filepath, sr=None, duration=1)  # Load only 1 sec to be fast
                    duration = librosa.get_duration(path=filepath)
                    
                    metadata_list.append({
                        'filename': filename,
                        # Standardize path separators for cross-platform compatibility
                        'filepath': filepath.replace('\\', '/'),
                        'genre': genre,
                        'duration': duration,
                        'sample_rate': sr
                    })
                except Exception as e:
                    # If a file is corrupted or unreadable, print a warning and skip it.
                    print(f"\nWARNING: Could not process {filepath}. Skipping file. Error: {e}")

    df = pd.DataFrame(metadata_list)
    return df


def get_mfcc_fingerprint(filepath: str, n_mfcc: int = 20) -> Optional[np.ndarray]:
    """
    Computes a stable 'fingerprint' of a track using the mean of its MFCCs.
    
    Args:
        filepath: Path to audio file
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Mean MFCC features or None if processing fails
    """
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' keys
        save_path: Optional path to save the plot
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    axs[0].plot(history["train_acc"], label="train_accuracy")
    axs[0].plot(history["val_acc"], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(history["train_loss"], label="train_error")
    axs[1].plot(history["val_loss"], label="val_error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, genres: List[str], 
                         title: str = "Confusion Matrix", 
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix array
        genres: List of genre names
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=genres, 
                yticklabels=genres, cmap='YlOrRd', cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

