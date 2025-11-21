from app.preprocess import create_combined_data
from app.recommender import build_tfidf_model, build_sbert_model

# Preprocess data and build models
df = create_combined_data()
build_tfidf_model(df)
build_sbert_model(df)
print("✅ Data preprocessing and model building completed.")

