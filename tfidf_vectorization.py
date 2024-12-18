from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_tfidf_matrix(texts):
    """Calculates the TF-IDF matrix for a list of texts."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts.values())
    return tfidf_matrix, vectorizer

def compute_cosine_similarity(tfidf_matrix):
    """Computes cosine similarity between resumes and job description based on their TF-IDF vectors."""
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def find_most_similar_resumes(similarity_matrix, top_n=3):
    """Finds the most similar resumes based on cosine similarity to the job description."""
    # Assume we are comparing each resume to the first one (index 0)
    sim_scores = list(enumerate(similarity_matrix[0]))  # First resume's similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:top_n+1]  # Exclude the job description itself