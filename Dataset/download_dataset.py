import pandas as pd
import os

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
    print("Balanced dataset shape:", balanced_df.shape)
    print("Balanced class distribution:\n", balanced_df['label'].value_counts())
    
    # Save dataset
    out_dir = r"E:\CS619\Graze\Prototype\graze_smasher\Dataset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fake_reviews_dataset.csv")
    balanced_df.to_csv(out_path, index=False)
    print(f"Dataset successfully saved to: {out_path}")
    
    # Display the first five records
    print("\nFirst five records:")
    print(balanced_df.head())

if __name__ == "__main__":
    download_and_prepare()
