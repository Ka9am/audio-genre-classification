"""
Unit tests for model definitions.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DeepBottleneckAE, EnhancedGenreClassifier, CRNN


def test_autoencoder_forward():
    """Test autoencoder forward pass."""
    model = DeepBottleneckAE(latent_dim=512)
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 130)  # Typical mel spectrogram shape
    
    output = model(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert not torch.isnan(output).any(), "Output should not contain NaN"


def test_autoencoder_encode():
    """Test autoencoder encoding."""
    model = DeepBottleneckAE(latent_dim=512)
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 130)
    
    latent = model.encode(x)
    
    assert latent.shape == (batch_size, 512), "Latent shape should be (batch, latent_dim)"


def test_classifier_forward():
    """Test classifier forward pass."""
    encoder = DeepBottleneckAE(latent_dim=512).encoder
    model = EnhancedGenreClassifier(encoder=encoder, latent_dim=512, num_genres=10)
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 130)
    
    output = model(x)
    
    assert output.shape == (batch_size, 10), "Output should be (batch, num_genres)"
    assert not torch.isnan(output).any(), "Output should not contain NaN"


def test_crnn_forward():
    """Test CRNN forward pass."""
    model = CRNN(num_genres=10)
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 130)
    
    output = model(x)
    
    assert output.shape == (batch_size, 10), "Output should be (batch, num_genres)"
    assert not torch.isnan(output).any(), "Output should not contain NaN"


def test_model_device_transfer():
    """Test model can be moved to CUDA if available."""
    if torch.cuda.is_available():
        model = DeepBottleneckAE(latent_dim=512)
        model = model.cuda()
        x = torch.randn(2, 1, 128, 130).cuda()
        
        output = model(x)
        assert output.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

