#!/usr/bin/env python3
"""Download and explore the MMLU-ProX-Lite dataset from Hugging Face."""

from datasets import load_dataset
import pandas as pd

def download_and_explore_dataset():
    """Download MMLU-ProX-Lite dataset and explore its structure."""
    print("Downloading MMLU-ProX-Lite dataset from Hugging Face...")
    
    # Load the dataset (using English config)
    dataset = load_dataset("li-lab/MMLU-ProX-Lite", "en")
    
    print(f"Dataset keys: {list(dataset.keys())}")
    
    # Explore each split
    for split_name, split_data in dataset.items():
        print(f"\n=== {split_name.upper()} SPLIT ===")
        print(f"Number of examples: {len(split_data)}")
        print(f"Features: {split_data.features}")
        
        # Show first few examples
        if len(split_data) > 0:
            df = pd.DataFrame(split_data[:3])
            print(f"\nFirst 3 examples:")
            for idx, row in df.iterrows():
                print(f"\nExample {idx + 1}:")
                for col, val in row.items():
                    print(f"  {col}: {val}")
    
    return dataset

if __name__ == "__main__":
    dataset = download_and_explore_dataset()