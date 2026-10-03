import os
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Global variables for model caching
_vectorizer = None
_model = None

# Ensure NLTK resources are available
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Applies the text preprocessing pipeline to a raw review string.
    Identical to the preprocessing used in ML model training.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase conversion
    text_lower = text.lower()
    
    # 2. Punctuation removal
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text_no_punc = text_lower.translate(translator)
    
    # 3. Tokenization
    tokens = word_tokenize(text_no_punc)
    
    # 4. Stop-word removal
    tokens_no_stop = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens_no_stop]
    
    # 6. Re-join tokens
    cleaned_text = " ".join(lemmatized_tokens)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def get_model_and_vectorizer():
    """
    Loads the trained SVM model and TF-IDF vectorizer.
    Caches them in memory to prevent reloading on subsequent requests.
    """
    global _vectorizer, _model
    
    if _vectorizer is None or _model is None:
        # Paths relative to the project directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "Trained_Model"))
        vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
        model_path = os.path.join(base_dir, "svm_model.pkl")
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model files not found in {base_dir}. Please run the ML model training script first!"
            )
            
        _vectorizer = joblib.load(vectorizer_path)
        _model = joblib.load(model_path)
        
    return _model, _vectorizer

def predict_review(review_text):
    """
    Takes a raw review string, preprocesses it, and predicts
    whether it is 'Genuine' or 'Fake' using the cached model.
    """
    if not review_text.strip():
        return "Invalid Input", ""
        
    # Preprocess text
    cleaned_text = preprocess_text(review_text)
    
    # Load model and vectorizer
    model, vectorizer = get_model_and_vectorizer()
    
    # Vectorize cleaned text
    text_vec = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    
    return prediction, cleaned_text
