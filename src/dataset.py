"""
Dataset classes for loading and preprocessing audio data.
"""

import os
import json
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa


class MelSpectrogramDataset(Dataset):
    """
    Dataset for loading pre-computed mel spectrograms from JSON file.
    """
    
    def __init__(self, json_path: str, normalize: bool = True):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to JSON file containing mel spectrograms
            normalize: Whether to normalize the spectrograms
        """
        with open(json_path, "r") as fp:
            data = json.load(fp)
        
        self.X = np.array(data["mel_spectrograms"])
        self.y = np.array(data["labels"])
        self.genres = data["mapping"]
        
        # Add channel dimension -> (num_samples, height, width, channels)
        if len(self.X.shape) == 3:
            self.X = self.X[..., np.newaxis]
        
        # Normalize if requested
        if normalize:
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min() + 1e-8)
        
        # Convert to PyTorch tensors (channels-first format)
        self.X = torch.tensor(self.X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(self.y, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (spectrogram, label)
        """
        return self.X[idx], self.y[idx]


class AudioDataset(Dataset):
    """
    Dataset for loading raw audio files and converting to mel spectrograms on-the-fly.
    """
    
    def __init__(self, filepaths: List[str], labels: List[int], 
                 sr: int = 22050, n_mels: int = 128, 
                 duration: Optional[float] = None, normalize: bool = True):
        """
        Initialize the dataset.
        
        Args:
            filepaths: List of paths to audio files
            labels: List of integer labels
            sr: Sample rate
            n_mels: Number of mel filter banks
            duration: Maximum duration in seconds (None for full audio)
            normalize: Whether to normalize the spectrograms
        """
        self.filepaths = filepaths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.normalize = normalize
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.filepaths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (mel_spectrogram, label)
        """
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            y, sr = librosa.load(filepath, sr=self.sr, duration=self.duration)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize if requested
            if self.normalize:
                mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (
                    mel_spec_db.max() - mel_spec_db.min() + 1e-8
                )
            
            # Add channel dimension and convert to tensor
            mel_spec_db = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
            
            return mel_spec_db, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return zeros as fallback
            return torch.zeros((1, self.n_mels, 100)), torch.tensor(label, dtype=torch.long)

