from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure to download the required resources for lemmatization
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by tokenizing, lowercasing, removing stopwords, and lemmatizing."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenization and lowercasing
    tokens = text.lower().split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)