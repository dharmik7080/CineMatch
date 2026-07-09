"""
CineMatch Project - Phase 1: Data Preprocessing & Feature Engineering
Sub-Phase 1.2: Metadata Parsing, Feature Combination, Lowercasing, and Stemming

Syllabus Reference:
- Unit 1.3: Data Transformation using .apply()
- Unit 3.2: Feature Engineering / Preprocessing (Text Normalization and Stemming)
"""

import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer

# Import loading and cleaning pipeline functions from our previous module for maximum reuse
from eda_and_cleaning import (
    load_datasets, 
    merge_movie_datasets, 
    clean_movie_dataframe, 
    clean_tv_dataframe, 
    standardize_tv_layout
)

# Initialize NLTK Porter Stemmer (Unit 3.2: Feature Preprocessing)
ps = PorterStemmer()

# TMDB TV Genre ID mapping to transform integer genre IDs into meaningful tag words
TV_GENRE_MAP = {
    10759: "Action & Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    10762: "Kids",
    9648: "Mystery",
    10763: "News",
    10764: "Reality",
    10765: "Sci-Fi & Fantasy",
    10766: "Soap",
    10767: "Talk",
    10768: "War & Politics",
    37: "Western"
}

def safe_parse_json(val, key_name='name', limit=None, job_filter=None):
    """
    Syllabus Reference: Unit 1.3 (Data Transformation) & Unit 3.2 (Feature Engineering)
    Safely parses a stringified JSON list of dictionaries using ast.literal_eval.
    Extracts the values under 'key_name' (e.g. genre name or actor name).
    Applies custom token cleaning by stripping spaces to prevent multi-word splitting
    (e.g., 'Science Fiction' -> 'ScienceFiction').
    """
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        data = ast.literal_eval(val)
        if not isinstance(data, list):
            return []
        
        extracted = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if job_filter:
                # E.g. Filter for director in crew list
                if item.get('job') == job_filter:
                    extracted.append(item.get(key_name, ''))
                    break  # Usually there is one main director
            else:
                extracted.append(item.get(key_name, ''))
                
        if limit:
            extracted = extracted[:limit]
            
        # Standardize strings: remove spaces to ensure names act as single distinct search tokens
        return [str(x).replace(" ", "") for x in extracted if x]
    except (ValueError, SyntaxError):
        return []

def extract_tv_genres(genre_ids_str):
    """
    Syllabus Reference: Unit 1.3 (Data Transformation) & Unit 3.2 (Feature Engineering)
    Parses TV Show genre IDs lists, maps them to standard genre names,
    and removes spaces/ampersands to create unified tag tokens.
    """
    if pd.isna(genre_ids_str) or not isinstance(genre_ids_str, str):
        return []
    try:
        ids = ast.literal_eval(genre_ids_str)
        if not isinstance(ids, list):
            return []
        
        genres_list = []
        for gid in ids:
            genre_name = TV_GENRE_MAP.get(gid)
            if genre_name:
                # E.g., "Sci-Fi & Fantasy" -> "Sci-FiFantasy"
                # E.g., "Action & Adventure" -> "ActionAdventure"
                cleaned = genre_name.replace(" & ", "").replace(" ", "")
                genres_list.append(cleaned)
        return genres_list
    except (ValueError, SyntaxError):
        return []

def stem_text(text):
    """
    Syllabus Reference: Unit 3.2 (Feature Preprocessing - Word Stemming)
    Reduces words to their linguistic root using NLTK's PorterStemmer.
    This step ensures related words map to the exact same feature token.
    """
    if not isinstance(text, str):
        return ""
    stemmed_words = [ps.stem(word) for word in text.split()]
    return " ".join(stemmed_words)

def main():
    print("======================================================================")
    print(" CineMatch Phase 1, Sub-Phase 1.2: Feature & Tag Engineering Path")
    print("======================================================================\n")
    
    # 1. Load the cleaned datasets from the previous step (Unit 1: Data Loading)
    print("--- [Step 1] Loading and Pre-cleaning Datasets ---")
    movies_raw, credits_raw, tv_raw = load_datasets()
    
    # Apply movie pipeline cleaning
    merged_movies = merge_movie_datasets(movies_raw, credits_raw)
    cleaned_movies, _, _, _, _ = clean_movie_dataframe(merged_movies)
    
    # Apply TV shows pipeline cleaning and schema standardization
    cleaned_tv, _, _, _, _ = clean_tv_dataframe(tv_raw)
    cleaned_tv = standardize_tv_layout(cleaned_tv)
    
    # 2. Movie DataFrame Feature Extraction (Unit 1.3 & Unit 3.2)
    print("--- [Step 2] Processing Metadata Arrays for Movies ---")
    
    # Transform list of dictionaries using .apply() to clean tokens
    cleaned_movies['genres'] = cleaned_movies['genres'].apply(lambda x: safe_parse_json(x))
    cleaned_movies['keywords'] = cleaned_movies['keywords'].apply(lambda x: safe_parse_json(x))
    cleaned_movies['cast'] = cleaned_movies['cast'].apply(lambda x: safe_parse_json(x, limit=3))
    cleaned_movies['crew'] = cleaned_movies['crew'].apply(lambda x: safe_parse_json(x, job_filter='Director'))
    
    # Tokenize overview string by splitting it into individual words
    cleaned_movies['overview'] = cleaned_movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    print("Movies metadata fields successfully parsed and cleaned.")
    
    # 3. TV Show DataFrame Feature Extraction (Unit 1.3 & Unit 3.2)
    print("\n--- [Step 3] Processing Genre Arrays for TV Shows ---")
    
    # Extract genres using mapping and split overview into word lists
    cleaned_tv['genres'] = cleaned_tv['genre_ids'].apply(extract_tv_genres)
    cleaned_tv['overview'] = cleaned_tv['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    print("TV Shows genre IDs successfully mapped and cleaned.")
    
    # 4. Create Unified Engineered 'tags' Column (Unit 3.2)
    print("\n--- [Step 4] Engineering Unified 'tags' Column ---")
    
    # Movies tags = overview + genres + keywords + cast + crew
    # Add lists of words/tokens together
    cleaned_movies['tags_list'] = (
        cleaned_movies['overview'] + 
        cleaned_movies['genres'] + 
        cleaned_movies['keywords'] + 
        cleaned_movies['cast'] + 
        cleaned_movies['crew']
    )
    # Join list of word tokens into a single space-separated string using .apply()
    cleaned_movies['tags'] = cleaned_movies['tags_list'].apply(lambda x: " ".join(x))
    
    # TV Shows tags = overview + genres * 4 + country_proxies * 3 + year_tag
    cleaned_tv['year_tag'] = cleaned_tv['first_air_date'].apply(lambda x: [x.split('-')[0]] if isinstance(x, str) and '-' in x else [])
    
    def get_country_proxies(countries_str):
        if pd.isna(countries_str) or not isinstance(countries_str, str):
            return []
        try:
            countries = ast.literal_eval(countries_str)
            return [f"{c}Network" for c in countries]
        except Exception:
            return []
    cleaned_tv['country_proxies'] = cleaned_tv['origin_country'].apply(get_country_proxies)

    cleaned_tv['tags_list'] = (
        cleaned_tv['overview'] + 
        cleaned_tv['genres'] * 4 +
        cleaned_tv['country_proxies'] * 3 +
        cleaned_tv['year_tag']
    )
    cleaned_tv['tags'] = cleaned_tv['tags_list'].apply(lambda x: " ".join(x))
    
    print("Unified 'tags' column engineered successfully for both domains.")
    
    # 5. Lowercasing tags (Unit 3.2 Text Normalization)
    print("\n--- [Step 5] Normalizing Tags to Lowercase ---")
    cleaned_movies['tags'] = cleaned_movies['tags'].apply(lambda x: x.lower())
    cleaned_tv['tags'] = cleaned_tv['tags'].apply(lambda x: x.lower())
    print("Lowercase normalization completed.")
    
    # 6. Apply Stemming using NLTK PorterStemmer (Unit 1.3 & Unit 3.2)
    print("\n--- [Step 6] Running Stemming with PorterStemmer ---")
    cleaned_movies['tags'] = cleaned_movies['tags'].apply(stem_text)
    cleaned_tv['tags'] = cleaned_tv['tags'].apply(stem_text)
    print("Stemming transformation successfully applied.")
    
    # Keep essential columns to form final data models, preserving popularity and dates
    movies_final = cleaned_movies[['movie_id', 'title', 'tags']]
    tv_final = cleaned_tv[['id', 'title', 'tags', 'popularity', 'first_air_date']]
    
    # Save the final engineered dataframes to CSV files
    movies_final.to_csv('engineered_movies.csv', index=False)
    tv_final.to_csv('engineered_tv_shows.csv', index=False)
    print("\nSaved final engineered datasets: 'engineered_movies.csv' and 'engineered_tv_shows.csv'.")
    
    # 7. Sneak Peek of Engineered Tags
    print("\n======================================================================")
    print(" SNEAK PEEK: ENGINEERED MOVIES TAGS (.head())")
    print("======================================================================")
    for idx, row in movies_final.head(3).iterrows():
        print(f"Movie Title: {row['title']}")
        print(f"Tags Snippet: {row['tags'][:150]}...\n")
        
    print("======================================================================")
    print(" SNEAK PEEK: ENGINEERED TV SHOWS TAGS (.head())")
    print("======================================================================")
    for idx, row in tv_final.head(3).iterrows():
        print(f"TV Show Title: {row['title']}")
        print(f"Tags Snippet: {row['tags'][:150]}...\n")
        
    print("Flight path successfully completed! Feature engineering is aligned and compliant.\n")

if __name__ == '__main__':
    main()
