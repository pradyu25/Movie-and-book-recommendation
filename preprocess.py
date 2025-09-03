import pandas as pd
import numpy as np
import ast
import re
import nltk
import json
import os
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def extract_names(json_str):
    try:
        items = ast.literal_eval(json_str)
        names = [i['name'] for i in items]
        return ' '.join(names)
    except:
        return ''

# Loaders
def load_movie_data(movies_path, credits_path):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
    movies['genres'] = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    movies['cast'] = movies['cast'].apply(lambda x: ' '.join([i['name'] for i in ast.literal_eval(x)[:3]]) if pd.notna(x) else '')
    movies['crew'] = movies['crew'].apply(lambda x: ' '.join([i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director']) if pd.notna(x) else '')
    movies['text'] = (
        movies['title_x'].fillna('') + ' ' +
        movies['genres'].fillna('') + ' ' +
        movies['overview'].fillna('') + ' ' +
        movies['keywords'].fillna('') + ' ' +
        movies['cast'].fillna('') + ' ' +
        movies['crew'].fillna('')
    )
    print("🔁 Preprocessing movie data...")
    movies['processed'] = movies['text'].apply(preprocess)
    movies = movies.rename(columns={'title_x': 'title'})
    movies['type'] = 'movie'
    movies = movies[['title', 'processed', 'type']]
    print(f"✅ Movies processed: {movies.shape}")
    return movies

def load_book_data(books_path, book_tags_path, to_read_path):
    books = pd.read_csv(books_path)
    book_tags = pd.read_csv(book_tags_path)
    to_read = pd.read_csv(to_read_path)

    popular_tags = book_tags['tag_id'].value_counts().head(500).index.tolist()
    filtered_tags = book_tags[book_tags['tag_id'].isin(popular_tags)]
    tag_map = filtered_tags.groupby('goodreads_book_id')['tag_id'].apply(lambda x: ' '.join(map(str, x))).reset_index()
    books = books.merge(tag_map, left_on='book_id', right_on='goodreads_book_id', how='left')
    read_counts = to_read['book_id'].value_counts().reset_index()
    read_counts.columns = ['book_id', 'read_count']
    books = books.merge(read_counts, on='book_id', how='left')
    books['read_count'] = books['read_count'].fillna(0).astype(int)
    description_text = books['description'].fillna('') if 'description' in books.columns else pd.Series([''] * len(books))
    books['text'] = (
        books['title'].fillna('') + ' ' +
        books['authors'].fillna('') + ' ' +
        books['original_title'].fillna('') + ' ' +
        description_text + ' ' +
        books['tag_id'].fillna('')
    )
    print("🔁 Preprocessing book data...")
    books['processed'] = books['text'].apply(preprocess)
    books['type'] = 'book'
    books = books[['title', 'processed', 'type']]
    print(f"✅ Books processed: {books.shape}")
    return books

# Image Cache
def load_image_cache(cache_path='data/combined_image_cache.json'):
    if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
        return {}
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Warning: Cache file is corrupted. Using empty cache.")
        return {}

def save_image_cache(cache, cache_path='data/combined_image_cache.json'):
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

# 🔍 API Keys
TMDB_API_KEY = "387b5701b6b0a44c01b587cbcffcc47d"  # Replace with your TMDB API key

# Image Fetching
def fetch_image_url(title, content_type):
    if content_type == 'movie':
        try:
            resp = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}")
            data = resp.json()
            if data.get("results"):
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except:
            return None
    elif content_type == 'book':
        try:
            resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}")
            data = resp.json()
            if data.get("items"):
                image_links = data["items"][0]["volumeInfo"].get("imageLinks", {})
                return image_links.get("thumbnail", None)
        except:
            return None
    return None

def fetch_and_cache_image_urls(df):
    cache = load_image_cache()
    new_entries = 0
    for _, row in df.iterrows():
        key = f"{row['title']}::{row['type']}"
        if key not in cache:
            url = fetch_image_url(row['title'], row['type'])
            cache[key] = url
            new_entries += 1
    print(f"🖼️  New image URLs fetched: {new_entries}")
    save_image_cache(cache)
    df['image_url'] = df.apply(lambda row: cache.get(f"{row['title']}::{row['type']}", None), axis=1)
    return df

# 🧠 Main
def create_combined_data():
    movies = load_movie_data('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
    books = load_book_data('data/books.csv', 'data/book_tags.csv', 'data/to_read.csv')
    combined_df = pd.concat([books, movies], ignore_index=True)
    combined_df.dropna(subset=['processed'], inplace=True)
    combined_df = combined_df[combined_df['processed'].str.strip() != '']
    print(f"✅ Combined text data shape: {combined_df.shape}")
    combined_df = fetch_and_cache_image_urls(combined_df)
    combined_df.to_csv('data/combined_books_movies.csv', index=False)
    combined_df[['title', 'type', 'image_url']].to_json('data/combined_books_movies.json', orient='records', indent=2)
    print("✅ Saved processed CSV and JSON.")
    return combined_df

if __name__ == "__main__":
    create_combined_data()

