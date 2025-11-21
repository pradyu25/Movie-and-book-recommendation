import streamlit as st
import pandas as pd
from recommender import load_data, load_models, recommend

# Load data and models
df = load_data()
tfidf, tfidf_matrix, sbert_embeddings = load_models()

# Streamlit UI
st.title(" Book & Movie Recommender")
st.write("Enter a book or movie title, or describe what you're looking for:")

query = st.text_input("Your search query:")

if query:
    results = recommend(query, df, tfidf, tfidf_matrix, sbert_embeddings)
    st.write(f"Top recommendations for: '{query}'")
    for idx, row in results.iterrows():
        st.write(f"**{row['title']}** ({row['type'].capitalize()}) ")

