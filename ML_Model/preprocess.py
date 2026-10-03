import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
# These resources are downloaded during initial environment setup
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    # Fallback downloads if they are missing
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Applies the text preprocessing pipeline to a raw review string:
    1. Lowercase conversion
    2. Punctuation removal
    3. Tokenization
    4. Stop-word removal
    5. Lemmatization
    
    Parameters:
    text (str): The raw input review text.
    
    Returns:
    str: The preprocessed and cleaned text.
    """
    # 0. Handle non-string or null inputs
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase conversion
    text_lower = text.lower()
    
    # 2. Punctuation removal
    # Replace punctuation with spaces to avoid joining words, then remove double spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text_no_punc = text_lower.translate(translator)
    
    # 3. Tokenization
    tokens = word_tokenize(text_no_punc)
    
    # 4. Stop-word removal
    tokens_no_stop = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization (reducing words to their dictionary base form)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens_no_stop]
    
    # 6. Re-join tokens back into a single string
    cleaned_text = " ".join(lemmatized_tokens)
    
    # Remove any extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# If executed directly, run preprocessing on a sample review
if __name__ == "__main__":
    sample_review = "This product is ABSOLUTELY amazing! I bought it for $19.99 and it's working flawlessly."
    processed = preprocess_text(sample_review)
    print("Original Review:")
    print(sample_review)
    print("\nProcessed Review:")
    print(processed)
