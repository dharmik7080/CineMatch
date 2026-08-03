"""
CineMatch Project - Phase 1: Data Preprocessing & Feature Vectorization
Sub-Phase 1.3: Text Vectorization and Unsupervised Metrics (Movies Vectorization)

Syllabus Reference:
- Unit 3.2: Text Feature Transformation (Bag of Words Model / Count Vectorizer)
- Unit 5: Unsupervised Metrics & Distance Measures (Cosine Similarity)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("======================================================================")
    print(" CineMatch Phase 1, Sub-Phase 1.3: Movie Vectorization & Similarity Path")
    print("======================================================================\n")
    
    # Step 1: Load the engineered Movies dataset
    print("--- [Step 1] Loading Engineered Movies Dataset ---")
    if not os.path.exists('engineered_movies.csv'):
        print("Error: 'engineered_movies.csv' not found. Run tag_engineering.py first.")
        return
    df_movies = pd.read_csv('engineered_movies.csv')
    print(f"Loaded dataset containing {df_movies.shape[0]} rows and {df_movies.shape[1]} columns.\n")
    
    # Handle missing tags if any (fill with empty string)
    df_movies['tags'] = df_movies['tags'].fillna('')
    
    # Step 2: Initialize CountVectorizer (Unit 3.2: Text Feature Transformation)
    # max_features=5000: Limits vocabulary to the 5,000 most frequent tokens.
    # stop_words='english': Excludes non-informative English articles, prepositions, etc.
    print("--- [Step 2] Initializing CountVectorizer (max_features=5000) ---")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    
    # Step 3: Transform tags into a feature coordinate matrix
    print("--- [Step 3] Transforming Tags into Feature Vectors ---")
    vectors = cv.fit_transform(df_movies['tags']).toarray()
    print(f"Generated Document-Term Matrix of shape: {vectors.shape} (Movies x Vocabulary)\n")
    
    # Step 4: Compute the Cosine Similarity Matrix (Unit 5: Unsupervised Similarity Metrics)
    print("--- [Step 4] Computing Pairwise Cosine Similarity Matrix ---")
    similarity = cosine_similarity(vectors)
    print(f"Cosine Similarity Matrix shape: {similarity.shape}\n")
    
    # Step 5: Create title-to-index map and dictionary representation
    movie_dict = df_movies.to_dict()
    
    # Step 6: Serialize and save outputs to pickle files in the models/ directory
    print("\n--- [Step 6] Serializing and Saving Pre-computed Artifacts ---")
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    movie_dict_path = os.path.join(models_dir, 'movie_dict.pkl')
    similarity_path = os.path.join(models_dir, 'similarity.pkl')
    
    # Save the dataframe dictionary representation
    with open(movie_dict_path, 'wb') as f:
        pickle.dump(movie_dict, f)
    print(f"--> Saved '{movie_dict_path}'")
    
    # Save the pre-computed similarity matrix array
    with open(similarity_path, 'wb') as f:
        pickle.dump(similarity, f)
    print(f"--> Saved '{similarity_path}'")
    
    print("\n======================================================================")
    print(" SUCCESS CONFIRMATION REPORT")
    print("======================================================================")
    print(f"Movies Count (N):            {similarity.shape[0]}")
    print(f"Similarity Matrix Shape:     {similarity.shape} (N x N)")
    print(f"Vocabulary Dimension (M):    {vectors.shape[1]}")
    print(f"Memory size of matrix array: {similarity.nbytes / (1024 ** 2):.2f} MB")
    print("======================================================================")
    print("Movie similarity vectorization pipeline completed successfully.\n")

if __name__ == '__main__':
    main()
