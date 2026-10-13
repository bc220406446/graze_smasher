"""
Graze Smasher Product Hunting Tool - Complete Model Training Pipeline
CS619 Final-Year Project

Trains and evaluates:
1. Calibrated Support Vector Machine (LinearSVC + CalibratedClassifierCV)
2. J-48 Equivalent Decision Tree (C4.5 / Entropy-based DecisionTreeClassifier)
3. Composite Ensemble (Weighted SVM + J48 + Heuristic Risk)

Performs 80/20 train/test evaluation and 5-fold Stratified Cross-Validation.
Saves comparison metrics to CSV and visual plots to Screenshots/.
Serializes models to Trained_Model/.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg') # Headless backend for headless server/CLI execution
import matplotlib.pyplot as plt
import seaborn as sns

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
    SCREENSHOTS_DIR,
    SVM_MODEL_PATH,
    VECTORIZER_PATH,
    J48_MODEL_PATH,
    WEIGHT_SVM,
    WEIGHT_J48,
    WEIGHT_RISK,
    J48_CRITERION,
    J48_MAX_DEPTH,
    J48_MIN_SAMPLES_LEAF,
    J48_RANDOM_STATE,
    SVM_C,
    SVM_RANDOM_STATE,
    J48_FEATURE_NAMES,
    IP_MAX_REVIEWS_PER_WINDOW,
)
from preprocess import preprocess_text
from feature_extraction import extract_features_df, extract_numeric_features
from ip_detector import generate_simulated_ips, IPVelocityTracker


def train_and_evaluate_all():
    print("=" * 70)
    print("   GRAZE SMASHER ML PIPELINE: SVM + J-48 + COMPOSITE EVALUATION")
    print("=" * 70)
    
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    
    # 1. Load Dataset
    print(f"\n[1/7] Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"      Dataset Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    
    # Add simulated IP column if missing
    if "simulated_ip" not in df.columns:
        print("      Generating clearly labelled simulated IP addresses for demo/testing...")
        df["simulated_ip"] = generate_simulated_ips(len(df))
        df.to_csv(DATASET_PATH, index=False)
        print("      Saved updated dataset with 'simulated_ip' column.")
        
    # Map target label: 'CG' -> 'Fake', 'OR' -> 'Genuine'
    df["label_mapped"] = df["label"].map({"CG": "Fake", "OR": "Genuine"})
    y_raw = df["label_mapped"]
    
    # 2. Text Preprocessing for SVM
    print("\n[2/7] Preprocessing text reviews for SVM...")
    df["text_"] = df["text_"].fillna("")
    df["clean_text"] = df["text_"].apply(preprocess_text)
    
    # 3. Numeric Feature Extraction for J-48
    print("\n[3/7] Extracting 8 numeric behavioral & sentiment features for J-48...")
    X_j48_df = extract_features_df(df, text_col="text_", rating_col="rating")
    print("      Features extracted:")
    for feat in J48_FEATURE_NAMES:
        print(f"       - {feat}: mean={X_j48_df[feat].mean():.3f}, std={X_j48_df[feat].std():.3f}")
        
    # 4. Train/Test Split (80/20 Stratified)
    print("\n[4/7] Splitting dataset: 80% Train, 20% Test (Stratified)...")
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
    
    # 5. Train Models
    # --- MODEL 1: Calibrated LinearSVC ---
    print("\n[5/7] Training Calibrated SVM (LinearSVC + CalibratedClassifierCV)...")
    base_svc = LinearSVC(C=SVM_C, random_state=SVM_RANDOM_STATE, dual=False)
    # CalibratedClassifierCV provides calibrated predict_proba() for LinearSVC
    svm_calibrated = CalibratedClassifierCV(estimator=base_svc, cv=5)
    svm_calibrated.fit(X_svm_train, y_train)
    
    # --- MODEL 2: J-48 Decision Tree ---
    print("      Training J-48 Decision Tree (C4.5 equivalent using entropy criterion)...")
    # Note: scikit-learn has no exact WEKA J48, so we use DecisionTreeClassifier
    # with criterion="entropy" (Information Gain) with max_depth and min_samples_leaf
    # to control tree pruning and prevent overfitting.
    j48_model = DecisionTreeClassifier(
        criterion=J48_CRITERION,
        max_depth=J48_MAX_DEPTH,
        min_samples_leaf=J48_MIN_SAMPLES_LEAF,
        random_state=J48_RANDOM_STATE
    )
    j48_model.fit(X_j48_train, y_train)
    
    # 6. Evaluation on Test Set (80/20)
    print("\n[6/7] Evaluating SVM, J-48, and Composite Ensemble on Test Set (20%)...")
    classes = list(svm_calibrated.classes_)
    fake_idx = classes.index("Fake")
    
    # Probabilities on test set
    svm_probs = svm_calibrated.predict_proba(X_svm_test)[:, fake_idx]
    j48_probs = j48_model.predict_proba(X_j48_test)[:, fake_idx]
    
    # Test predictions for standalone models
    svm_preds = svm_calibrated.predict(X_svm_test)
    j48_preds = j48_model.predict(X_j48_test)
    
    # IP Velocity & Mismatch risk for Composite Model
    test_ips = df["simulated_ip"].iloc[test_idx].values
    tracker = IPVelocityTracker()
    test_ip_flags = []
    for ip in test_ips:
        flagged, _ = tracker.check_ip(ip, record=True)
        test_ip_flags.append(1.0 if flagged else 0.0)
    test_ip_flags = np.array(test_ip_flags)
    
    test_mismatch_risks = np.minimum(1.0, X_j48_test["rating_vs_sentiment_mismatch"].values / 2.0)
    test_heuristic_risks = (0.6 * test_ip_flags) + (0.4 * test_mismatch_risks)
    
    # Composite fake probability formula:
    composite_fake_probs = (
        (WEIGHT_SVM * svm_probs) +
        (WEIGHT_J48 * j48_probs) +
        (WEIGHT_RISK * test_heuristic_risks)
    )
    composite_preds = np.where(composite_fake_probs >= 0.50, "Fake", "Genuine")
    
    # Calculate performance metrics
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        return acc, prec, rec, f1

    svm_acc, svm_prec, svm_rec, svm_f1 = get_metrics(y_test, svm_preds)
    j48_acc, j48_prec, j48_rec, j48_f1 = get_metrics(y_test, j48_preds)
    comp_acc, comp_prec, comp_rec, comp_f1 = get_metrics(y_test, composite_preds)
    
    # 5-Fold Cross Validation
    print("      Running 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_svm_scores = []
    cv_j48_scores = []
    cv_comp_scores = []
    
    all_texts = df["clean_text"].values
    all_X_j48 = X_j48_df
    
    for fold, (trn_i, val_i) in enumerate(skf.split(all_texts, y_raw), 1):
        # SVM Fold: fit vectorizer only on training fold to prevent leakage
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
        
        # Composite Fold
        fold_mismatch = np.minimum(1.0, all_X_j48["rating_vs_sentiment_mismatch"].iloc[val_i].values / 2.0)
        fold_risk = 0.4 * fold_mismatch # Baseline mismatch risk
        fold_comp_prob = (WEIGHT_SVM * val_svm_prob) + (WEIGHT_J48 * val_j48_prob) + (WEIGHT_RISK * fold_risk)
        fold_comp_pred = np.where(fold_comp_prob >= 0.50, "Fake", "Genuine")
        cv_comp_scores.append(accuracy_score(y_raw.iloc[val_i], fold_comp_pred))

    cv_svm_mean = np.mean(cv_svm_scores)
    cv_j48_mean = np.mean(cv_j48_scores)
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
            "Model": "Composite Ensemble (SVM + J48 + Risk)",
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
    csv_out_path = os.path.join(SCREENSHOTS_DIR, "model_comparison_table.csv")
    comparison_df.to_csv(csv_out_path, index=False)
    print(f"\nSaved comparison table to: {csv_out_path}")
    
    # Confusion Matrices
    cm_svm = confusion_matrix(y_test, svm_preds, labels=["Genuine", "Fake"])
    cm_j48 = confusion_matrix(y_test, j48_preds, labels=["Genuine", "Fake"])
    cm_comp = confusion_matrix(y_test, composite_preds, labels=["Genuine", "Fake"])
    
    print("\nConfusion Matrix - SVM Alone:")
    print(pd.DataFrame(cm_svm, index=["Actual Genuine", "Actual Fake"], columns=["Pred Genuine", "Pred Fake"]))
    
    print("\nConfusion Matrix - J-48 Alone:")
    print(pd.DataFrame(cm_j48, index=["Actual Genuine", "Actual Fake"], columns=["Pred Genuine", "Pred Fake"]))
    
    print("\nConfusion Matrix - Composite Ensemble:")
    print(pd.DataFrame(cm_comp, index=["Actual Genuine", "Actual Fake"], columns=["Pred Genuine", "Pred Fake"]))
    
    # 7. Generate Visualizations
    print("\n[7/7] Generating visualization plots in Screenshots/...")
    
    # Plot 1: Performance Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics_melted = comparison_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score (%)"
    )
    sns.barplot(data=metrics_melted, x="Metric", y="Score (%)", hue="Model", palette="viridis")
    plt.title("Model Performance Comparison: SVM vs J-48 vs Composite", fontsize=14, pad=12)
    plt.ylim(50, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    metrics_plot_path = os.path.join(SCREENSHOTS_DIR, "model_metrics.png")
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()
    print(f"      Saved: {metrics_plot_path}")
    
    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["Genuine", "Fake"]
    
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=labels, yticklabels=labels)
    axes[0].set_title(f"SVM Alone\n(Accuracy: {svm_acc*100:.1f}%)")
    axes[0].set_ylabel("Actual Label")
    axes[0].set_xlabel("Predicted Label")
    
    sns.heatmap(cm_j48, annot=True, fmt="d", cmap="Greens", ax=axes[1], xticklabels=labels, yticklabels=labels)
    axes[1].set_title(f"J-48 Decision Tree Alone\n(Accuracy: {j48_acc*100:.1f}%)")
    axes[1].set_ylabel("Actual Label")
    axes[1].set_xlabel("Predicted Label")
    
    sns.heatmap(cm_comp, annot=True, fmt="d", cmap="Purples", ax=axes[2], xticklabels=labels, yticklabels=labels)
    axes[2].set_title(f"Composite Ensemble\n(Accuracy: {comp_acc*100:.1f}%)")
    axes[2].set_ylabel("Actual Label")
    axes[2].set_xlabel("Predicted Label")
    
    plt.suptitle("Confusion Matrix Comparison (Salminen et al. Test Set)", fontsize=16, y=1.03)
    plt.tight_layout()
    cm_plot_path = os.path.join(SCREENSHOTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {cm_plot_path}")
    
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

