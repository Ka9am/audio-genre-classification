# Music Genre Classification

A deep learning project for classifying music genres using convolutional autoencoders and neural networks. This project implements a two-phase training approach: first training an autoencoder to learn compressed audio representations, then training a classifier on top of the learned features.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project implements a music genre classification system using:
- **Deep Bottleneck Autoencoder**: Learns compressed representations of mel spectrograms
- **Enhanced Classifier**: Multi-layer classifier with batch normalization and dropout
- **Two-Phase Training**: Pre-training encoder, then fine-tuning the full model

The model achieves ~76% accuracy on the GTZAN dataset (10 genres).

##  Features

- Modular codebase with clear separation of concerns
- Reproducible experiments with fixed random seeds
- Comprehensive configuration system (YAML)
- One-command training and evaluation (Makefile)
- Unit tests for key components
- Automated data downloader
- Visualization tools for results

##  Project Structure

```
.
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── models.py          # Model definitions
│   ├── dataset.py         # Dataset classes
│   ├── train.py           # Training functions
│   └── utils.py           # Utility functions
├── configs/               # Configuration files
│   └── default_config.yaml
├── scripts/               # Executable scripts
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── download_data.py  # Data downloader
├── tests/                # Unit tests
│   ├── test_models.py
│   └── test_utils.py
├── notebooks/           # Jupyter notebooks (original exploration)
│   └── genre_classification.ipynb
├── results/             # Output directory
│   ├── checkpoints/     # Model checkpoints
│   ├── plots/           # Visualization plots
│   ├── logs/            # Training logs
│   └── metrics/         # Evaluation metrics
├── Data/                # Dataset directory
├── requirements.txt      # Python dependencies
├── environment.yml      # Conda environment
├── Makefile            # One-command scripts
└── README.md           # This file
```

##  Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Installation

#### Option 1: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd MGC_DL

# Install dependencies
make setup
# or
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate genre_classification
```

##  Data

### GTZAN Dataset

The project uses the GTZAN Genre Collection dataset, which contains 1000 audio files across 10 genres:
- Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

**Note**: The GTZAN dataset is not publicly available for direct download due to copyright restrictions. You need to:
1. Visit [Marsyas Downloads](http://marsyas.info/downloads/datasets.html)
2. Request access to the GTZAN Genre Collection
3. Download and extract to `Data/genres_original/`

### Sample Dataset

For testing, you can create a small sample dataset:

```bash
python scripts/download_data.py --create-sample --samples 10
```

This creates a sample dataset with 10 files per genre in `Data/genres_original_sample/`.

### Data Preparation

The project expects pre-processed mel spectrograms in JSON format (`data_segments.json`). If you need to generate this from raw audio, refer to the notebook `notebooks/genre_classification.ipynb`.

##  Usage

### Quick Start

```bash
# 1. Setup environment
make setup

# 2. Download/prepare data
make download-data

# 3. Train the model
make train

# 4. Evaluate the model
make evaluate
```

### Training

Train the full model (autoencoder + classifier):

```bash
# Using Makefile
make train

# Or directly
python scripts/train.py --config configs/default_config.yaml
```

To skip autoencoder training and use an existing checkpoint:

```bash
python scripts/train.py --config configs/default_config.yaml --skip-ae
```

### Evaluation

Evaluate a trained model:

```bash
# Using Makefile
make evaluate

# Or directly
python scripts/evaluate.py \
    --config configs/default_config.yaml \
    --checkpoint results/checkpoints/best_classifier.pth
```

### Testing

Run unit tests:

```bash
make test
# or
pytest tests/ -v
```

##  Results

The model achieves the following performance on the GTZAN test set:

- **Test Accuracy**: ~76.3%
- **Best Validation Accuracy (Phase 1)**: ~70.1%
- **Best Validation Accuracy (Phase 2)**: ~79.4%

Results are saved in:
- `results/metrics/evaluation_results.json` - Detailed metrics
- `results/plots/confusion_matrix.png` - Confusion matrix
- `results/plots/training_history_combined.png` - Training curves

##  Configuration

All hyperparameters are configured in `configs/default_config.yaml`:

```yaml
seed: 42  # Random seed for reproducibility

model:
  latent_dim: 512
  num_genres: 10

training:
  batch_size: 32
  ae:
    epochs: 50
    learning_rate: 0.001
  phase1:
    epochs: 30
    learning_rate: 0.001
  phase2:
    epochs: 50
    learning_rate: 0.00005
```

##  Testing

The project includes unit tests for key components:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models.py -v
pytest tests/test_utils.py -v
```

##  Code Quality

Format and lint the code:

```bash
# Format code
make format

# Lint code
make lint
```

##  Model Architecture

### Autoencoder
- **Encoder**: 4 convolutional blocks with batch normalization
- **Bottleneck**: Fully connected layer to latent dimension (512)
- **Decoder**: 4 transposed convolutional blocks

### Classifier
- **Feature Extractor**: Pre-trained encoder from autoencoder
- **Classifier Head**: 4 fully connected layers with batch norm and dropout
- **Output**: 10 classes (one per genre)

##  Training Process

1. **Phase 0**: Train autoencoder to reconstruct mel spectrograms
2. **Phase 1**: Train classifier with frozen encoder
3. **Phase 2**: Fine-tune entire model (unfreeze encoder)

##  Model Checkpoints

Pre-trained model checkpoints are saved in `results/checkpoints/`:
- `best_autoencoder.pth` - Trained autoencoder
- `best_classifier.pth` - Trained classifier (after phase 2)

##  Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

##  License

This project is for educational purposes. The GTZAN dataset has specific usage restrictions - please refer to the original dataset license.

##  Acknowledgments

- GTZAN Genre Collection dataset
- PyTorch community
- Librosa for audio processing

##  GitHub Release

To create a tagged release:

```bash
# Prepare release (creates tag v1.0.0)
bash scripts/prepare_release.sh v1.0.0

# Or manually:
git tag -a v1.0.0 -m "Release v1.0.0: Music Genre Classification"
git push origin v1.0.0
```

Then create a release on GitHub and attach the tag.

##  Contact

For questions or issues, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: 2025

