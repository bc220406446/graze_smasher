import os
from pathlib import Path
import pandas as pd

def download_and_prepare():
    url = "https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv"
    print("Downloading dataset from:", url)
    
    # Read dataset
    df = pd.read_csv(url)
    print("Original dataset shape:", df.shape)
    print("Original class distribution:\n", df['label'].value_counts())
    
    # Slice a balanced subset of 10,000 reviews (5,000 CG and 5,000 OR)
    cg_subset = df[df['label'] == 'CG'].head(5000)
    or_subset = df[df['label'] == 'OR'].head(5000)
    
    balanced_df = pd.concat([cg_subset, or_subset]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Retain strictly the 4 authentic text/rating/category/label columns (no simulated_ip)
    cols_to_keep = ['category', 'rating', 'label', 'text_']
    balanced_df = balanced_df[[c for c in cols_to_keep if c in balanced_df.columns]]
    
    print("Balanced dataset shape:", balanced_df.shape)
    print("Balanced class distribution:\n", balanced_df['label'].value_counts())
    
    # Save dataset to local Dataset folder
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "fake_reviews_dataset.csv"
    balanced_df.to_csv(out_path, index=False)
    print(f"Dataset successfully saved to: {out_path}")
    
    # Display the first five records
    print("\nFirst five records:")
    print(balanced_df.head())

if __name__ == "__main__":
    download_and_prepare()
