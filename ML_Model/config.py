"""
Graze Smasher Product Hunting Tool - Configuration
CS619 Final-Year Project

Centralized configuration for Machine Learning models, weights, thresholds,
and detection heuristics. Avoids hardcoding any parameters across the codebase.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Dataset" / "fake_reviews_dataset.csv"
TRAINED_MODEL_DIR = BASE_DIR / "Trained_Model"
MODEL_COMPARISON_CSV = TRAINED_MODEL_DIR / "model_comparison_table.csv"

# Model Artifact Paths
SVM_MODEL_PATH = TRAINED_MODEL_DIR / "svm_model.pkl"
VECTORIZER_PATH = TRAINED_MODEL_DIR / "vectorizer.pkl"
J48_MODEL_PATH = TRAINED_MODEL_DIR / "j48_model.pkl"

# Composite Authenticity Score Weights (100% Text-Based ML, Must sum to 1.0)
# Formula: fake_risk = (WEIGHT_SVM * svm_fake_prob) + (WEIGHT_J48 * j48_fake_prob) + (WEIGHT_SENTIMENT * sentiment_mismatch_risk)
# authenticity_score = (1.0 - fake_risk) * 100
WEIGHT_SVM = 0.50
WEIGHT_J48 = 0.30
WEIGHT_SENTIMENT = 0.20
WEIGHT_RISK = WEIGHT_SENTIMENT  # Backwards compatibility alias

# Binary Classification Threshold (Authenticity Score 0 - 100, higher = more genuine)
# Strict Binary Classification (matching dataset OR vs CG):
# Score >= 50.0: Genuine | Score < 50.0: Fake
AUTHENTICITY_GENUINE_THRESHOLD = 50.0



# Review Submission Rate Limiting (Spam & Throttling Protection)
IP_MAX_REVIEWS_PER_WINDOW = 3
IP_TIME_WINDOW_MINUTES = 1

# J-48 (C4.5 Decision Tree) Hyperparameters
# Note: scikit-learn doesn't have WEKA's J48 directly, so we use DecisionTreeClassifier
# with criterion="entropy" (Information Gain) as the standard C4.5 / J48 equivalent.
J48_CRITERION = "entropy"
J48_MAX_DEPTH = 8
J48_MIN_SAMPLES_LEAF = 5
J48_RANDOM_STATE = 42

# SVM Hyperparameters
SVM_C = 1.0
SVM_RANDOM_STATE = 42

# Feature Column Names for J-48
J48_FEATURE_NAMES = [
    "review_length",
    "word_count",
    "exclamation_count",
    "uppercase_ratio",
    "repeated_word_ratio",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "rating_vs_sentiment_mismatch",
]

# Automated Alert Thresholds & Analytics Settings
ALERT_FAKE_RATIO_THRESHOLD = 0.50  # Alert triggered when product fake review ratio >= 50%
ALERT_COOLDOWN_HOURS = 24          # Suppression cooldown for product ratio alerts (hours); reviewer alerts have no cooldown

