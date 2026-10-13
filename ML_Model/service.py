"""
Graze Smasher Product Hunting Tool - Unified Analysis Service
CS619 Final-Year Project

Provides the single unified service function `analyze_review(...)` used by the
entire system (web interface, REST API, background tasks, tests).
Loads models once and caches them in memory.
"""

import os
import threading
import pandas as pd
import numpy as np
import joblib

try:
    from .config import (
        SVM_MODEL_PATH,
        VECTORIZER_PATH,
        J48_MODEL_PATH,
        WEIGHT_SVM,
        WEIGHT_J48,
        WEIGHT_RISK,
        AUTHENTICITY_GENUINE_THRESHOLD,
        AUTHENTICITY_SUSPICIOUS_THRESHOLD,
        J48_FEATURE_NAMES,
    )
    from .preprocess import preprocess_text
    from .feature_extraction import extract_numeric_features
    from .ip_detector import check_ip_velocity
except ImportError:
    from config import (
        SVM_MODEL_PATH,
        VECTORIZER_PATH,
        J48_MODEL_PATH,
        WEIGHT_SVM,
        WEIGHT_J48,
        WEIGHT_RISK,
        AUTHENTICITY_GENUINE_THRESHOLD,
        AUTHENTICITY_SUSPICIOUS_THRESHOLD,
        J48_FEATURE_NAMES,
    )
    from preprocess import preprocess_text
    from feature_extraction import extract_numeric_features
    from ip_detector import check_ip_velocity

# Cached model instances
_svm_model = None
_vectorizer = None
_j48_model = None
_model_lock = threading.Lock()


def load_models():
    """
    Loads serialized ML models into memory once and caches them.
    Thread-safe implementation.
    """
    global _svm_model, _vectorizer, _j48_model
    
    if _svm_model is not None and _vectorizer is not None and _j48_model is not None:
        return _svm_model, _vectorizer, _j48_model
        
    with _model_lock:
        if _svm_model is None or _vectorizer is None or _j48_model is None:
            if not os.path.exists(SVM_MODEL_PATH):
                raise FileNotFoundError(f"SVM model not found at {SVM_MODEL_PATH}. Run train_all.py first!")
            if not os.path.exists(VECTORIZER_PATH):
                raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Run train_all.py first!")
            if not os.path.exists(J48_MODEL_PATH):
                raise FileNotFoundError(f"J-48 model not found at {J48_MODEL_PATH}. Run train_all.py first!")
                
            _svm_model = joblib.load(SVM_MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            _j48_model = joblib.load(J48_MODEL_PATH)
            
    return _svm_model, _vectorizer, _j48_model


def get_fake_probability(model, X) -> float:
    """
    Helper to extract the probability of the 'Fake' class from an sklearn classifier.
    Supports CalibratedClassifierCV and DecisionTreeClassifier.
    """
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        classes = list(model.classes_)
        if "Fake" in classes:
            fake_idx = classes.index("Fake")
            return float(probas[fake_idx])
        elif 1 in classes:
            fake_idx = classes.index(1)
            return float(probas[fake_idx])
        else:
            return float(probas[-1])
    elif hasattr(model, "decision_function"):
        # Sigmoid fallback if probability calibration is not present
        score = model.decision_function(X)[0]
        # Sigmoid: 1 / (1 + e^-score)
        prob = 1.0 / (1.0 + np.exp(-score))
        return float(prob)
    else:
        pred = model.predict(X)[0]
        return 1.0 if pred in ["Fake", 1] else 0.0


def compute_authenticity(
    svm_fake_prob: float,
    j48_fake_prob: float,
    ip_flag: bool,
    mismatch_score: float
) -> tuple[float, str]:
    """
    Computes the composite authenticity score (0 - 100, higher = more genuine)
    and categorical label (Genuine / Suspicious / Fake).
    
    Composite fake risk formula:
        fake_risk = (w_svm * svm_fake) + (w_j48 * j48_fake) + (w_risk * risk_score)
        authenticity_score = (1.0 - fake_risk) * 100
    """
    # Risk component combining IP velocity and rating-sentiment mismatch
    ip_risk = 1.0 if ip_flag else 0.0
    # mismatch_score is typically between 0.0 and 2.0; normalize to [0.0, 1.0]
    mismatch_risk = min(1.0, mismatch_score / 2.0)
    
    # Combined heuristic risk
    risk_score = (0.6 * ip_risk) + (0.4 * mismatch_risk)
    
    # Total fake risk [0.0, 1.0]
    fake_risk = (
        (WEIGHT_SVM * svm_fake_prob) +
        (WEIGHT_J48 * j48_fake_prob) +
        (WEIGHT_RISK * risk_score)
    )
    fake_risk = max(0.0, min(1.0, fake_risk))
    
    # Authenticity score (higher = genuine)
    authenticity_score = round((1.0 - fake_risk) * 100.0, 2)
    
    # Threshold labeling
    if authenticity_score >= AUTHENTICITY_GENUINE_THRESHOLD:
        label = "Genuine"
    elif authenticity_score >= AUTHENTICITY_SUSPICIOUS_THRESHOLD:
        label = "Suspicious"
    else:
        label = "Fake"
        
    return authenticity_score, label


def analyze_review(
    text: str,
    rating: int | float | None = None,
    ip: str | None = None,
    record_ip: bool = True
) -> dict:
    """
    Single unified service function used across the entire application:
    1. Preprocesses text and extracts features
    2. Runs Calibrated SVM for TF-IDF fake probability
    3. Runs J-48 (Decision Tree) on 8 numeric features
    4. Evaluates sentiment polarity and rating mismatch
    5. Evaluates IP velocity rule
    6. Calculates composite authenticity score and label
    
    Parameters:
        text (str): Raw review text
        rating (int | float | None): Star rating (1-5) or None
        ip (str | None): Client IP address or None
        record_ip (bool): Whether this check registers as an IP submission
        
    Returns:
        dict: {
            "svm_score": float,            # SVM Fake probability [0.0, 1.0]
            "j48_score": float,            # J-48 Fake probability [0.0, 1.0]
            "sentiment_polarity": float,   # TextBlob polarity [-1.0, 1.0]
            "sentiment_subjectivity": float,
            "ip_flag": bool,               # True if IP exceeded velocity limit
            "final_score": float,          # Authenticity score [0.0, 100.0]
            "label": str,                  # "Genuine" | "Suspicious" | "Fake"
            "cleaned_text": str,           # Lemmatized text
            "mismatch_score": float
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "svm_score": 0.0,
            "j48_score": 0.0,
            "sentiment_polarity": 0.0,
            "sentiment_subjectivity": 0.0,
            "ip_flag": False,
            "final_score": 0.0,
            "label": "Invalid Input",
            "cleaned_text": "",
            "mismatch_score": 0.0,
        }
        
    # 1. Load models
    svm_model, vectorizer, j48_model = load_models()
    
    # 2. Text preprocessing & TF-IDF
    cleaned_text = preprocess_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    
    # 3. Model 1: Calibrated SVM Fake Probability
    svm_score = round(get_fake_probability(svm_model, text_vec), 4)
    
    # 4. Feature Extraction & Model 2: J-48 Decision Tree
    features = extract_numeric_features(text, rating)
    features_df = pd.DataFrame([features], columns=J48_FEATURE_NAMES)
    j48_score = round(get_fake_probability(j48_model, features_df), 4)
    
    # 5. Sentiment & IP velocity rule
    sentiment_polarity = round(features["sentiment_polarity"], 4)
    sentiment_subjectivity = round(features["sentiment_subjectivity"], 4)
    mismatch_score = round(features["rating_vs_sentiment_mismatch"], 4)
    
    ip_flag, ip_reason = check_ip_velocity(ip, record=record_ip)
    
    # 6. Composite Score & Label
    final_score, label = compute_authenticity(
        svm_fake_prob=svm_score,
        j48_fake_prob=j48_score,
        ip_flag=ip_flag,
        mismatch_score=mismatch_score
    )
    
    return {
        "svm_score": svm_score,
        "j48_score": j48_score,
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
        "ip_flag": ip_flag,
        "final_score": final_score,
        "label": label,
        "cleaned_text": cleaned_text,
        "mismatch_score": mismatch_score,
        "ip_reason": ip_reason,
    }

