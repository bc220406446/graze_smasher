import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib

# Import preprocess function
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_text

def train_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "Dataset", "fake_reviews_dataset.csv"))
    model_dir = os.path.abspath(os.path.join(current_dir, "..", "Trained_Model"))
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading dataset from:", dataset_path)
    df = pd.read_csv(dataset_path)
    
    print("Dataset shape:", df.shape)
    print("Preprocessing review texts...")
    
    # Preprocess all reviews
    # Fill NA values in text_ with empty string before processing
    df['text_'] = df['text_'].fillna('')
    df['clean_text'] = df['text_'].apply(preprocess_text)
    
    # Map label to binary if needed, but keeping text categories is fine. 
    # Let's map 'CG' to 'Fake' and 'OR' to 'Genuine' for clear reporting.
    df['label_mapped'] = df['label'].map({'CG': 'Fake', 'OR': 'Genuine'})
    
    X = df['clean_text']
    y = df['label_mapped']
    
    print("Splitting dataset into 80% Train and 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print("Vectorizing text using TF-IDF Vectorizer...")
    # Initialize TF-IDF Vectorizer
    # min_df=2 to remove words that appear only once, sublinear_tf=True scales term frequency logarithmically
    vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature dimensions: {X_train_vec.shape[1]}")
    
    print("Training Support Vector Machine (LinearSVC)...")
    # Train LinearSVC (Support Vector Classification)
    # C=1.0 is standard regularization strength, dual=False is preferred when n_samples > n_features
    svm_model = LinearSVC(C=1.0, random_state=42, dual=False)
    svm_model.fit(X_train_vec, y_train)
    
    print("Evaluating model performance on test set...")
    y_pred = svm_model.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    # Extract precision, recall, and f1 for the 'Fake' class specifically (or weighted average)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print("\n" + "="*40)
    print("             MODEL METRICS")
    print("="*40)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['Genuine', 'Fake'])
    print(pd.DataFrame(cm, index=['Actual Genuine', 'Actual Fake'], columns=['Predicted Genuine', 'Predicted Fake']))
    
    # Save the vectorizer and trained model
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    model_path = os.path.join(model_dir, "svm_model.pkl")
    
    print(f"\nSaving vectorizer to: {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Saving trained SVM model to: {model_path}")
    joblib.dump(svm_model, model_path)
    
    print("\nPipeline training complete!")

if __name__ == "__main__":
    train_pipeline()
