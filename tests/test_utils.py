"""
Unit tests for utility functions.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed


def test_set_seed():
    """Test that set_seed produces reproducible results."""
    set_seed(42)
    np_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()
    
    set_seed(42)
    np_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()
    
    assert np_val1 == np_val2, "NumPy random should be reproducible"
    assert torch_val1 == torch_val2, "PyTorch random should be reproducible"


def test_set_seed_different_seeds():
    """Test that different seeds produce different results."""
    set_seed(42)
    val1 = np.random.rand()
    
    set_seed(123)
    val2 = np.random.rand()
    
    assert val1 != val2, "Different seeds should produce different values"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

