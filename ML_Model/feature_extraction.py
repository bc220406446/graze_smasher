"""
Graze Smasher Product Hunting Tool - Feature Extraction for J-48
CS619 Final-Year Project

Extracts numeric, non-textual behavioral and linguistic features for the
J-48 (C4.5 Decision Tree) classifier.
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob

try:
    from .config import J48_FEATURE_NAMES
except ImportError:
    from config import J48_FEATURE_NAMES


def compute_sentiment(text: str) -> tuple[float, float]:
    """
    Computes pre-trained sentiment polarity and subjectivity using TextBlob.
    
    Returns:
        (polarity, subjectivity):
            polarity: float in [-1.0, 1.0] (negative to positive)
            subjectivity: float in [0.0, 1.0] (objective to subjective)
    """
    if not text or not isinstance(text, str):
        return 0.0, 0.0
    blob = TextBlob(text)
    return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)


def extract_numeric_features(text: str, rating: float | int | None = None) -> dict[str, float]:
    """
    Extracts the 8 numerical features required by the J-48 classifier:
    1. review_length: Total character count of raw text
    2. word_count: Total word count
    3. exclamation_count: Total exclamation mark count ('!')
    4. uppercase_ratio: Proportion of uppercase characters
    5. repeated_word_ratio: Proportion of duplicate words (vocabulary repetition)
    6. sentiment_polarity: TextBlob polarity [-1.0, 1.0]
    7. sentiment_subjectivity: TextBlob subjectivity [0.0, 1.0]
    8. rating_vs_sentiment_mismatch: Absolute discrepancy between star rating and sentiment
    
    Parameters:
        text (str): Raw review text
        rating (float | int | None): Star rating (1 to 5) or None
        
    Returns:
        dict: Mapping of feature names to float values.
    """
    if not isinstance(text, str):
        text = ""
        
    char_len = len(text)
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    excl_count = text.count('!')
    
    # Uppercase ratio
    uppercase_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_chars / max(char_len, 1)
    
    # Repeated word ratio: (total_words - unique_words) / total_words
    if word_count > 0:
        unique_words = len(set(words))
        repeated_word_ratio = (word_count - unique_words) / word_count
    else:
        repeated_word_ratio = 0.0
        
    # Sentiment analysis via TextBlob
    polarity, subjectivity = compute_sentiment(text)
    
    # Rating vs Sentiment Mismatch:
    # Star rating is on [1.0, 5.0] scale. Normalize to [-1.0, 1.0]: (rating - 3) / 2
    # If rating is None or NaN, default to neutral (3.0 -> 0.0)
    if rating is None or pd.isna(rating):
        norm_rating = 0.0
    else:
        try:
            r_val = float(rating)
            # Clamp between 1.0 and 5.0
            r_val = max(1.0, min(5.0, r_val))
            norm_rating = (r_val - 3.0) / 2.0
        except (ValueError, TypeError):
            norm_rating = 0.0
            
    mismatch = abs(norm_rating - polarity)
    
    return {
        "review_length": float(char_len),
        "word_count": float(word_count),
        "exclamation_count": float(excl_count),
        "uppercase_ratio": float(uppercase_ratio),
        "repeated_word_ratio": float(repeated_word_ratio),
        "sentiment_polarity": float(polarity),
        "sentiment_subjectivity": float(subjectivity),
        "rating_vs_sentiment_mismatch": float(mismatch),
    }


def extract_features_df(df: pd.DataFrame, text_col: str = "text_", rating_col: str = "rating") -> pd.DataFrame:
    """
    Extracts numerical features for an entire pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        text_col (str): Column name containing review text
        rating_col (str): Column name containing star ratings
        
    Returns:
        pd.DataFrame: DataFrame containing the 8 numerical features.
    """
    feature_rows = []
    has_rating = rating_col in df.columns
    
    for _, row in df.iterrows():
        t = row[text_col] if text_col in df.columns else ""
        r = row[rating_col] if has_rating else None
        feature_rows.append(extract_numeric_features(t, r))
        
    return pd.DataFrame(feature_rows, columns=J48_FEATURE_NAMES)

