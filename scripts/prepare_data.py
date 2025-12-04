#!/usr/bin/env python3
"""
Script to prepare data segments JSON from audio files.
This processes audio files and creates mel spectrograms.
"""

import os
import json
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed


def process_audio_file(filepath: str, sr: int = 22050, n_mels: int = 128, 
                       duration: float = 30.0) -> np.ndarray:
    """
    Process an audio file into a mel spectrogram.
    
    Args:
        filepath: Path to audio file
        sr: Sample rate
        n_mels: Number of mel filter banks
        duration: Duration to load (seconds)
        
    Returns:
        Mel spectrogram as numpy array
    """
    try:
        y, sr = librosa.load(filepath, sr=sr, duration=duration)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )
        
        return mel_spec_db.T  # Transpose to (time, frequency)
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def create_data_segments(dataset_path: str, output_path: str, 
                        segment_duration: float = 3.0) -> None:
    """
    Create data segments JSON from audio files.
    
    Args:
        dataset_path: Path to genres_original directory
        output_path: Path to save JSON file
        segment_duration: Duration of each segment in seconds
    """
    print(f"Processing audio files from {dataset_path}...")
    
    genres = sorted([d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))])
    
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    
    mel_spectrograms = []
    labels = []
    
    for genre in tqdm(genres, desc="Processing genres"):
        genre_path = os.path.join(dataset_path, genre)
        files = sorted([f for f in os.listdir(genre_path) if f.endswith('.wav')])
        
        for filename in tqdm(files, desc=f"  {genre}", leave=False):
            filepath = os.path.join(genre_path, filename)
            
            # Process full audio file
            mel_spec = process_audio_file(filepath)
            
            if mel_spec is not None:
                mel_spectrograms.append(mel_spec.tolist())
                labels.append(genre_to_idx[genre])
    
    # Save to JSON
    data = {
        "mel_spectrograms": mel_spectrograms,
        "labels": labels,
        "mapping": genres
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"\nData segments saved to {output_path}")
    print(f"Total samples: {len(mel_spectrograms)}")
    print(f"Genres: {genres}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data segments JSON")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Data/genres_original",
        help="Path to genres_original directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_segments.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory {args.dataset} does not exist.")
        print("Please download the GTZAN dataset first.")
        return
    
    create_data_segments(args.dataset, args.output)
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()

