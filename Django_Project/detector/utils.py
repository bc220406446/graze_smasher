"""
Graze Smasher Product Hunting Tool - Detector Utility Service
CS619 Final-Year Project

Provides the unified review analysis pipeline to Django views, APIs,
and management commands. Reuses and exposes `ML_Model.service.analyze_review`.
"""

import os
import sys

# Add project root and ML_Model to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DJANGO_ROOT = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(DJANGO_ROOT)
ML_MODEL_DIR = os.path.join(WORKSPACE_ROOT, "ML_Model")

for path in [WORKSPACE_ROOT, ML_MODEL_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ML_Model.service import (
    analyze_review as ml_analyze_review,
    load_models,
    compute_authenticity,
)
from ML_Model.preprocess import preprocess_text
from ML_Model.feature_extraction import extract_numeric_features
from ML_Model import config


def analyze_review(
    text: str,
    rating: int | float | None = None,
    ip: str | None = None,
    record_ip: bool = True
) -> dict:
    """
    Primary service function for review authenticity analysis:
    - Textual probability via Calibrated SVM (LinearSVC)
    - Behavioral/structural probability via J-48 (C4.5 Decision Tree)
    - Lexical Sentiment Polarity via TextBlob
    - Suspicious Velocity Detection via Rule-Based IP sliding window
    - Composite Authenticity Score (0 - 100, higher = more genuine)
    - Classification Label: Genuine, Suspicious, or Fake

    Returns:
        dict: {
            "svm_score": float,          # Fake probability [0.0, 1.0]
            "j48_score": float,          # Fake probability [0.0, 1.0]
            "sentiment_polarity": float, # Polarity [-1.0, 1.0]
            "ip_flag": bool,             # True if IP velocity exceeded
            "final_score": float,        # Authenticity score [0.0, 100.0]
            "label": str,                # "Genuine" | "Suspicious" | "Fake"
            "cleaned_text": str,
            "sentiment_subjectivity": float,
            "mismatch_score": float,
            "ip_reason": str
        }
    """
    return ml_analyze_review(text=text, rating=rating, ip=ip, record_ip=record_ip)


def predict_review(review_text: str, rating: int | float | None = None, ip: str | None = None) -> tuple[str, str]:
    """
    Backwards-compatible wrapper returning (label, cleaned_text).
    Maintains compatibility with prototype callers while routing through
    the unified analyze_review() pipeline.
    """
    if not review_text or not review_text.strip():
        return "Invalid Input", ""
        
    result = analyze_review(text=review_text, rating=rating, ip=ip, record_ip=False)
    return result["label"], result["cleaned_text"]
