"""
Graze Smasher Product Hunting Tool - Complete Model Training Pipeline
CS619 Final-Year Project

Trains and evaluates:
1. Calibrated Support Vector Machine (LinearSVC + CalibratedClassifierCV)
2. J-48 Equivalent Decision Tree (C4.5 / Entropy-based DecisionTreeClassifier)
3. TextBlob NLP Sentiment Engine & Rating Mismatch
4. Composite Ensemble (Weighted SVM 50% + J-48 30% + TextBlob Sentiment 20%)

Performs 80/20 train/test evaluation and 5-fold Stratified Cross-Validation.
Saves comparison metrics to CSV and visual plots to Trained_Model/.
Serializes models to Trained_Model/.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Ensure ML_Model directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    DATASET_PATH,
    TRAINED_MODEL_DIR,
    MODEL_COMPARISON_CSV,
    SVM_MODEL_PATH,
    VECTORIZER_PATH,
    J48_MODEL_PATH,
    WEIGHT_SVM,
    WEIGHT_J48,
    WEIGHT_SENTIMENT,
    J48_CRITERION,
    J48_MAX_DEPTH,
    J48_MIN_SAMPLES_LEAF,
    J48_RANDOM_STATE,
    SVM_C,
    SVM_RANDOM_STATE,
    J48_FEATURE_NAMES,
)
from preprocess import preprocess_text
from feature_extraction import extract_features_df, extract_numeric_features


def train_and_evaluate_all():
    print("=" * 70)
    print("   GRAZE SMASHER ML PIPELINE: SVM + J-48 + TEXTBLOB COMPOSITE")
    print("=" * 70)
    
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
    
    # 1. Load Dataset
    print(f"\n[1/6] Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"      Dataset Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    
    # Map target label: 'CG' -> 'Fake', 'OR' -> 'Genuine'
    df["label_mapped"] = df["label"].map({"CG": "Fake", "OR": "Genuine"})
    y_raw = df["label_mapped"]
    
    # 2. Text Preprocessing for SVM
    print("\n[2/6] Preprocessing text reviews for SVM...")
    df["text_"] = df["text_"].fillna("")
    df["clean_text"] = df["text_"].apply(preprocess_text)
    
    # 3. Numeric Feature Extraction for J-48 & Sentiment Mismatch
    print("\n[3/6] Extracting 8 numeric behavioral & sentiment features for J-48...")
    X_j48_df = extract_features_df(df, text_col="text_", rating_col="rating")
    print("      Features extracted:")
    for feat in J48_FEATURE_NAMES:
        print(f"       - {feat}: mean={X_j48_df[feat].mean():.3f}, std={X_j48_df[feat].std():.3f}")
        
    # 4. Train/Test Split (80/20 Stratified)
    print("\n[4/6] Splitting dataset: 80% Train, 20% Test (Stratified)...")
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y_raw
    )
    
    train_texts = df["clean_text"].iloc[train_idx]
    test_texts = df["clean_text"].iloc[test_idx]
    
    X_j48_train = X_j48_df.iloc[train_idx]
    X_j48_test = X_j48_df.iloc[test_idx]
    
    y_train = y_raw.iloc[train_idx]
    y_test = y_raw.iloc[test_idx]
    
    # Text Vectorization
    print("      Fitting TF-IDF Vectorizer (ngram=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))
    X_svm_train = vectorizer.fit_transform(train_texts)
    X_svm_test = vectorizer.transform(test_texts)
    print(f"      TF-IDF Vocabulary Size: {X_svm_train.shape[1]}")
    
    # --- MODEL 1: Calibrated LinearSVC ---
    print("\n[5/6] Training Models...")
    print("      Training Calibrated SVM (LinearSVC + CalibratedClassifierCV)...")
    base_svc = LinearSVC(C=SVM_C, random_state=SVM_RANDOM_STATE, dual=False)
    svm_calibrated = CalibratedClassifierCV(estimator=base_svc, cv=5)
    svm_calibrated.fit(X_svm_train, y_train)
    
    # --- MODEL 2: J-48 Decision Tree ---
    print("      Training J-48 Decision Tree (C4.5 equivalent using entropy criterion)...")
    j48_model = DecisionTreeClassifier(
        criterion=J48_CRITERION,
        max_depth=J48_MAX_DEPTH,
        min_samples_leaf=J48_MIN_SAMPLES_LEAF,
        random_state=J48_RANDOM_STATE
    )
    j48_model.fit(X_j48_train, y_train)
    
    # 6. Evaluation on Test Set (80/20)
    print("\n[6/6] Evaluating SVM, J-48, TextBlob Sentiment, and Composite Ensemble...")
    classes = list(svm_calibrated.classes_)
    fake_idx = classes.index("Fake")
    
    # Probabilities on test set
    svm_probs = svm_calibrated.predict_proba(X_svm_test)[:, fake_idx]
    j48_probs = j48_model.predict_proba(X_j48_test)[:, fake_idx]
    
    # Standalone model predictions
    svm_preds = svm_calibrated.predict(X_svm_test)
    j48_preds = j48_model.predict(X_j48_test)
    
    # TextBlob Sentiment & Rating Mismatch Risk (Normalized into [0.0, 1.0])
    test_mismatch_risks = np.minimum(1.0, np.maximum(0.0, X_j48_test["rating_vs_sentiment_mismatch"].values / 2.0))
    tb_preds = np.where(test_mismatch_risks >= 0.50, "Fake", "Genuine")
    
    # Composite fake probability formula (50% SVM + 30% J-48 + 20% TextBlob Sentiment):
    composite_fake_probs = (
        (WEIGHT_SVM * svm_probs) +
        (WEIGHT_J48 * j48_probs) +
        (WEIGHT_SENTIMENT * test_mismatch_risks)
    )
    composite_preds = np.where(composite_fake_probs >= 0.50, "Fake", "Genuine")
    
    # Calculate performance metrics helper
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        return acc, prec, rec, f1

    svm_acc, svm_prec, svm_rec, svm_f1 = get_metrics(y_test, svm_preds)
    j48_acc, j48_prec, j48_rec, j48_f1 = get_metrics(y_test, j48_preds)
    tb_acc, tb_prec, tb_rec, tb_f1 = get_metrics(y_test, tb_preds)
    comp_acc, comp_prec, comp_rec, comp_f1 = get_metrics(y_test, composite_preds)
    
    # 5-Fold Stratified Cross Validation
    print("      Running 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_svm_scores = []
    cv_j48_scores = []
    cv_tb_scores = []
    cv_comp_scores = []
    
    all_texts = df["clean_text"].values
    all_X_j48 = X_j48_df
    all_mismatch = np.minimum(1.0, np.maximum(0.0, all_X_j48["rating_vs_sentiment_mismatch"].values / 2.0))
    
    for fold, (trn_i, val_i) in enumerate(skf.split(all_texts, y_raw), 1):
        # SVM Fold
        cv_vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))
        cv_X_svm_train = cv_vectorizer.fit_transform(all_texts[trn_i])
        cv_X_svm_val = cv_vectorizer.transform(all_texts[val_i])
        
        clf_svm = CalibratedClassifierCV(
            estimator=LinearSVC(C=SVM_C, random_state=SVM_RANDOM_STATE, dual=False), cv=3
        )
        clf_svm.fit(cv_X_svm_train, y_raw.iloc[trn_i])
        val_svm_pred = clf_svm.predict(cv_X_svm_val)
        val_svm_prob = clf_svm.predict_proba(cv_X_svm_val)[:, fake_idx]
        cv_svm_scores.append(accuracy_score(y_raw.iloc[val_i], val_svm_pred))
        
        # J-48 Fold
        clf_j48 = DecisionTreeClassifier(
            criterion=J48_CRITERION,
            max_depth=J48_MAX_DEPTH,
            min_samples_leaf=J48_MIN_SAMPLES_LEAF,
            random_state=J48_RANDOM_STATE
        )
        clf_j48.fit(all_X_j48.iloc[trn_i], y_raw.iloc[trn_i])
        val_j48_pred = clf_j48.predict(all_X_j48.iloc[val_i])
        val_j48_prob = clf_j48.predict_proba(all_X_j48.iloc[val_i])[:, fake_idx]
        cv_j48_scores.append(accuracy_score(y_raw.iloc[val_i], val_j48_pred))
        
        # TextBlob Sentiment Fold
        fold_mismatch = all_mismatch[val_i]
        val_tb_pred = np.where(fold_mismatch >= 0.50, "Fake", "Genuine")
        cv_tb_scores.append(accuracy_score(y_raw.iloc[val_i], val_tb_pred))
        
        # Composite Fold
        fold_comp_prob = (WEIGHT_SVM * val_svm_prob) + (WEIGHT_J48 * val_j48_prob) + (WEIGHT_SENTIMENT * fold_mismatch)
        fold_comp_pred = np.where(fold_comp_prob >= 0.50, "Fake", "Genuine")
        cv_comp_scores.append(accuracy_score(y_raw.iloc[val_i], fold_comp_pred))

    cv_svm_mean = np.mean(cv_svm_scores)
    cv_j48_mean = np.mean(cv_j48_scores)
    cv_tb_mean = np.mean(cv_tb_scores)
    cv_comp_mean = np.mean(cv_comp_scores)
    
    # Summary Table
    comparison_df = pd.DataFrame([
        {
            "Model": "Calibrated SVM (LinearSVC)",
            "Accuracy": round(svm_acc * 100, 2),
            "Precision": round(svm_prec * 100, 2),
            "Recall": round(svm_rec * 100, 2),
            "F1-Score": round(svm_f1 * 100, 2),
            "5-Fold CV Accuracy": round(cv_svm_mean * 100, 2),
        },
        {
            "Model": "J-48 Decision Tree (C4.5 Entropy)",
            "Accuracy": round(j48_acc * 100, 2),
            "Precision": round(j48_prec * 100, 2),
            "Recall": round(j48_rec * 100, 2),
            "F1-Score": round(j48_f1 * 100, 2),
            "5-Fold CV Accuracy": round(cv_j48_mean * 100, 2),
        },
        {
            "Model": "TextBlob Sentiment Engine",
            "Accuracy": round(tb_acc * 100, 2),
            "Precision": round(tb_prec * 100, 2),
            "Recall": round(tb_rec * 100, 2),
            "F1-Score": round(tb_f1 * 100, 2),
            "5-Fold CV Accuracy": round(cv_tb_mean * 100, 2),
        },
        {
            "Model": "Composite Ensemble (SVM + J48 + Sentiment)",
            "Accuracy": round(comp_acc * 100, 2),
            "Precision": round(comp_prec * 100, 2),
            "Recall": round(comp_rec * 100, 2),
            "F1-Score": round(comp_f1 * 100, 2),
            "5-Fold CV Accuracy": round(cv_comp_mean * 100, 2),
        },
    ])
    
    print("\n" + "=" * 70)
    print("                    MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70)
    
    # Save CSV comparison table
    csv_out_path = str(MODEL_COMPARISON_CSV)
    comparison_df.to_csv(csv_out_path, index=False)
    print(f"\nSaved comparison table to: {csv_out_path}")
    
    # Confusion Matrices
    cm_svm = confusion_matrix(y_test, svm_preds, labels=["Genuine", "Fake"])
    cm_j48 = confusion_matrix(y_test, j48_preds, labels=["Genuine", "Fake"])
    cm_tb = confusion_matrix(y_test, tb_preds, labels=["Genuine", "Fake"])
    cm_comp = confusion_matrix(y_test, composite_preds, labels=["Genuine", "Fake"])
    
    # Print Confusion Matrices to Console (No PNG output)
    print("\nConfusion Matrices (Actual vs Predicted on Test Set):")
    labels = ["Genuine", "Fake"]
    print("1. Calibrated SVM:")
    print(pd.DataFrame(cm_svm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))
    print("\n2. J-48 Decision Tree:")
    print(pd.DataFrame(cm_j48, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))
    print("\n3. TextBlob Sentiment Engine:")
    print(pd.DataFrame(cm_tb, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))
    print("\n4. Composite Ensemble:")
    print(pd.DataFrame(cm_comp, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))
    
    # Save Model Artifacts
    print(f"\nSerializing trained models to: {TRAINED_MODEL_DIR}")
    joblib.dump(svm_calibrated, SVM_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(j48_model, J48_MODEL_PATH)
    print(f" - Saved SVM Model:     {SVM_MODEL_PATH}")
    print(f" - Saved TF-IDF Vocab:  {VECTORIZER_PATH}")
    print(f" - Saved J-48 Tree:     {J48_MODEL_PATH}")
    
    print("\nTraining and Evaluation pipeline finished successfully!")


if __name__ == "__main__":
    train_and_evaluate_all()
