def preprocess_text(text):
    """Preprocess text by tokenizing, lowercasing, and removing stopwords."""
    # Tokenization and lowercasing
    tokens = text.lower().split()
    # Remove stopwords (this could be improved with a more sophisticated stopword list)
    stopwords = set(['the', 'is', 'in', 'and', 'a', 'to', 'of'])
    tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(tokens)