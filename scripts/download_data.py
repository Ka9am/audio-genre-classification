#!/usr/bin/env python3
"""
Script to download GTZAN dataset.
Note: GTZAN dataset is not publicly available for direct download due to copyright.
This script provides instructions and a sample dataset creation utility.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Optional


def create_sample_dataset(source_dir: str, dest_dir: str, samples_per_genre: int = 10) -> None:
    """
    Create a small sample dataset from the full dataset for testing.
    
    Args:
        source_dir: Path to full dataset
        dest_dir: Path to save sample dataset
        samples_per_genre: Number of samples per genre to include
    """
    print(f"Creating sample dataset with {samples_per_genre} samples per genre...")
    
    genres = sorted([d for d in os.listdir(source_dir) 
                    if os.path.isdir(os.path.join(source_dir, d))])
    
    os.makedirs(dest_dir, exist_ok=True)
    
    for genre in genres:
        genre_source = os.path.join(source_dir, genre)
        genre_dest = os.path.join(dest_dir, genre)
        os.makedirs(genre_dest, exist_ok=True)
        
        files = sorted([f for f in os.listdir(genre_source) if f.endswith('.wav')])
        sample_files = files[:samples_per_genre]
        
        for file in sample_files:
            src = os.path.join(genre_source, file)
            dst = os.path.join(genre_dest, file)
            shutil.copy2(src, dst)
        
        print(f"  Copied {len(sample_files)} files from {genre}")
    
    print(f"\nSample dataset created at {dest_dir}")


def print_download_instructions() -> None:
    """Print instructions for downloading the GTZAN dataset."""
    print("=" * 70)
    print("GTZAN Dataset Download Instructions")
    print("=" * 70)
    print("\nThe GTZAN dataset is not publicly available for direct download")
    print("due to copyright restrictions. Please follow these steps:\n")
    print("1. Visit: http://marsyas.info/downloads/datasets.html")
    print("2. Request access to the GTZAN Genre Collection")
    print("3. Download the dataset and extract it to Data/genres_original/")
    print("\nAlternatively, you can use a sample dataset for testing:")
    print("  python scripts/download_data.py --create-sample --samples 10")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download or prepare GTZAN dataset")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset from existing data"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per genre for sample dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="Data/genres_original",
        help="Source directory for sample dataset"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="Data/genres_original_sample",
        help="Destination directory for sample dataset"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        if not os.path.exists(args.source):
            print(f"Error: Source directory {args.source} does not exist.")
            print("Please download the GTZAN dataset first.")
            print_download_instructions()
            return
        
        create_sample_dataset(args.source, args.dest, args.samples)
    else:
        print_download_instructions()


if __name__ == "__main__":
    main()

