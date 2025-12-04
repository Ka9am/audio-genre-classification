.PHONY: help install train evaluate test clean download-data setup format lint

# Default target
help:
	@echo "Music Genre Classification - Makefile Commands"
	@echo "=============================================="
	@echo "setup          - Set up the environment (install dependencies)"
	@echo "download-data  - Download or prepare dataset"
	@echo "train          - Train the model"
	@echo "evaluate       - Evaluate the trained model"
	@echo "test           - Run unit tests"
	@echo "format         - Format code with black"
	@echo "lint           - Lint code with flake8"
	@echo "clean          - Clean generated files"
	@echo ""

# Setup environment
setup:
	@echo "Setting up environment..."
	pip install -r requirements.txt
	@echo "Environment setup complete!"

# Download data
download-data:
	@echo "Preparing dataset..."
	python scripts/download_data.py
	@echo "Dataset preparation complete!"

# Train model
train:
	@echo "Training model..."
	python scripts/train.py --config configs/default_config.yaml
	@echo "Training complete!"

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	python scripts/evaluate.py --config configs/default_config.yaml
	@echo "Evaluation complete!"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "Tests complete!"

# Format code
format:
	@echo "Formatting code..."
	python -m black src/ scripts/ tests/
	@echo "Formatting complete!"

# Lint code
lint:
	@echo "Linting code..."
	python -m flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503
	@echo "Linting complete!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf .ipynb_checkpoints/
	@echo "Clean complete!"

