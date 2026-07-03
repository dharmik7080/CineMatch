"""
CineMatch Project - Phase 1: Data Preprocessing & Feature Vectorization
Sub-Phase 1.3: Text Vectorization and Unsupervised Metrics (TV Shows Vectorization)

Syllabus Reference:
- Unit 3.2: Text Feature Transformation (Bag of Words Model / Count Vectorizer)
- Unit 5: Unsupervised Metrics & Distance Measures (Cosine Similarity)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("======================================================================")
    print(" CineMatch Phase 1, Sub-Phase 1.3: Vectorization & Similarity Path")
    print("======================================================================\n")
    
    # Step 1: Load the engineered TV shows dataset (Unit 1: Data Loading)
    print("--- [Step 1] Loading Engineered TV Shows Dataset ---")
    df_tv = pd.read_csv('engineered_tv_shows.csv')
    print(f"Loaded dataset containing {df_tv.shape[0]} rows and {df_tv.shape[1]} columns.\n")
    
    # Handle missing tags if any (fill with empty string)
    df_tv['tags'] = df_tv['tags'].fillna('')
    
    # Step 2: Initialize CountVectorizer (Unit 3.2: Text Feature Transformation)
    # Mathematical Concept: Bag of Words (BoW) Representation
    # - Converts a collection of text documents to a matrix of token counts.
    # - max_features=5000: Limits the vocabulary to the 5,000 most frequent word tokens.
    # - stop_words='english': Excludes non-informative English articles, prepositions, etc. (e.g. 'the', 'is', 'and').
    print("--- [Step 2] Initializing CountVectorizer (max_features=5000) ---")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    
    # Step 3: Transform tags into a feature coordinate matrix (Unit 3.2: Vector Mapping)
    # Mathematical Concept: Document-Term Matrix (DTM)
    # - Let D = {d1, d2, ..., dN} be the set of N TV shows (documents).
    # - Let V = {w1, w2, ..., wM} be our vocabulary of size M = 5000 unique word stems.
    # - The vectorizer maps each show d_i to a high-dimensional vector x_i in R^5000.
    # - The coordinate x_i,j represents the frequency of occurrence of word stem w_j in document d_i.
    print("--- [Step 3] Transforming Tags into Feature Vectors ---")
    vectors = cv.fit_transform(df_tv['tags']).toarray()
    print(f"Generated Document-Term Matrix of shape: {vectors.shape} (Shows x Vocabulary)\n")
    
    # Step 4: Compute the Cosine Similarity Matrix (Unit 5: Unsupervised Similarity Metrics)
    # Mathematical Concept: Cosine Similarity
    # - Measures the cosine of the angle theta between two document vectors, x_a and x_b.
    # - It evaluates directional alignment rather than length/magnitude of the frequency vectors.
    # - Formula:
    #       Similarity(x_a, x_b) = cos(theta) = (x_a . x_b) / (||x_a|| * ||x_b||)
    #                            = [sum_{i=1}^M (x_a,i * x_b,i)] / [sqrt(sum_{i=1}^M (x_a,i)^2) * sqrt(sum_{i=1}^M (x_b,i)^2)]
    # - Output similarity matrix S is an N x N matrix, where S_a,b in [0, 1] represents the similarity between show a and show b.
    print("--- [Step 4] Computing Pairwise Cosine Similarity Matrix ---")
    similarity = cosine_similarity(vectors)
    print(f"Cosine Similarity Matrix shape: {similarity.shape}\n")
    
    # Step 5: Create Title-to-Index Map and DataFrame Dictionary
    print("--- [Step 5] Creating Title-to-Index Maps ---")
    # Mapping dictionary to easily locate a TV show's index by title
    title_to_index = pd.Series(df_tv.index, index=df_tv['title']).to_dict()
    print(f"Successfully generated mapping dictionary for {len(title_to_index)} titles.")
    
    # Prepare the DataFrame dictionary representation (matches movie recommender schema)
    tv_dict = df_tv.to_dict()
    
    # Step 6: Serialize and save outputs to pickle files (Unit 3.2 & Production Serialization)
    print("\n--- [Step 6] Serializing and Saving Pre-computed Artifacts ---")
    # Save the dataframe dictionary representation
    with open('tv_dict.pkl', 'wb') as f:
        pickle.dump(tv_dict, f)
    print("--> Saved 'tv_dict.pkl' for movie recommender compatibility.")
    
    # Save the pre-computed similarity matrix array
    with open('tv_similarity.pkl', 'wb') as f:
        pickle.dump(similarity, f)
    print("--> Saved 'tv_similarity.pkl' containing similarity matrix.")
    
    # Step 7: Print terminal confirmation and shape details
    print("\n======================================================================")
    print(" SUCCESS CONFIRMATION REPORT")
    print("======================================================================")
    print(f"TV Shows Count (N):          {similarity.shape[0]}")
    print(f"Similarity Matrix Shape:     {similarity.shape} (N x N)")
    print(f"Vocabulary Dimension (M):    {vectors.shape[1]}")
    print(f"Similarity Matrix Datatype:  {similarity.dtype}")
    print(f"Memory size of matrix array: {similarity.nbytes / (1024 ** 2):.2f} MB")
    print("======================================================================")
    print("Flight path successfully completed! TV domain similarity is vectorized.\n")

if __name__ == '__main__':
    main()
