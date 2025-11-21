import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# Load combined data
def load_data():
    return pd.read_csv('data/combined_books_movies.csv')

# Build and save TF-IDF model
def build_tfidf_model(df):
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['processed'])
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    np.save('models/tfidf_matrix.npy', tfidf_matrix.toarray())

# Build and save SBERT embeddings
def build_sbert_model(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['processed'].tolist(), show_progress_bar=True)
    np.save('models/sbert_embeddings.npy', embeddings)

# Load models
def load_models():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    tfidf_matrix = np.load('models/tfidf_matrix.npy')
    sbert_embeddings = np.load('models/sbert_embeddings.npy')
    return tfidf, tfidf_matrix, sbert_embeddings

# Recommend items based on query
def recommend(query, df, tfidf, tfidf_matrix, sbert_embeddings, top_n=10):
    # Determine if query is keyword-based or semantic
    if len(query.split()) <= 3:
        # Keyword-based: use TF-IDF
        query_vec = tfidf.transform([query]).toarray()
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    else:
        # Semantic: use SBERT
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, sbert_embeddings)[0]
    # Get top N recommendations
    indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[indices][['title', 'type']].copy()
    results['score'] = similarities[indices]
    return results

