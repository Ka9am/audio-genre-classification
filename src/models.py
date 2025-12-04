"""
PyTorch model definitions for music genre classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DeepBottleneckAE(nn.Module):
    """
    Deep Bottleneck Autoencoder for learning compressed audio representations.
    
    Architecture:
    - Encoder: Convolutional layers with batch norm and ReLU
    - Bottleneck: Fully connected layer to latent dimension
    - Decoder: Transposed convolutions to reconstruct input
    """
    
    def __init__(self, latent_dim: int = 512):
        """
        Initialize the autoencoder.
        
        Args:
            latent_dim: Dimension of the latent space
        """
        super(DeepBottleneckAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 2, 2)),
            
            # First deconv block
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Second deconv block
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third deconv block
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Fourth deconv block
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width)
            
        Returns:
            Reconstructed tensor
        """
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        decoded = self.decoder(latent)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        return latent


class EnhancedGenreClassifier(nn.Module):
    """
    Enhanced classifier with pre-trained encoder and deeper classification head.
    Uses a pre-trained autoencoder encoder for feature extraction.
    """
    
    def __init__(self, encoder: nn.Module, latent_dim: int = 512, num_genres: int = 10):
        """
        Initialize the classifier.
        
        Args:
            encoder: Pre-trained encoder from autoencoder
            latent_dim: Dimension of latent space
            num_genres: Number of genre classes
        """
        super(EnhancedGenreClassifier, self).__init__()
        
        # Freeze encoder initially (will be unfrozen during fine-tuning)
        self.encoder = encoder
        
        # Enhanced classifier head with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_genres)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width)
            
        Returns:
            Class logits of shape (batch, num_genres)
        """
        # Get latent representation from encoder
        with torch.set_grad_enabled(self.training):
            latent = self.encoder(x)
        
        # Classify from latent space
        logits = self.classifier(latent)
        return logits


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for music genre classification.
    Combines CNN for feature extraction with RNN for temporal modeling.
    """
    
    def __init__(self, num_genres: int = 10):
        """
        Initialize the CRNN model.
        
        Args:
            num_genres: Number of genre classes
        """
        super(CRNN, self).__init__()
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # RNN layers
        self.rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_genres)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CRNN.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width)
            
        Returns:
            Class logits
        """
        # CNN feature extraction
        x = self.conv_layers(x)  # (batch, 128, h', w')
        
        # Reshape for RNN: (batch, channels, height, width) -> (batch, height, channels*width)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, height, channels * width)
        
        # RNN
        rnn_out, _ = self.rnn(x)
        
        # Use last time step
        last_out = rnn_out[:, -1, :]
        
        # Classify
        logits = self.classifier(last_out)
        return logits

