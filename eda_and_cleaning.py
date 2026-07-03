"""
CineMatch Project - Phase 1: Data Preprocessing & Exploratory Data Analysis (EDA)
Sub-Phase 1.1: Initial Dataset Loading, Structure Auditing, Merging, and Preprocessing

Syllabus Reference: Unit 1 - Data Analysis with Pandas & EDA
- Loading tabular data (pd.read_csv)
- Structural properties exploration (.shape, .info(), .dtypes, memory_usage)
- Merging and combining DataFrames (pd.merge)
- Cleaning and Preprocessing (dropna, fillna, drop_duplicates, filtering empty strings)
- Restructuring and column standardization (rename)
"""

import pandas as pd
import numpy as np

def load_datasets():
    """
    Syllabus Topic: Data Loading
    Loads the datasets using pd.read_csv() as required by our EDA checklist.
    """
    print("--- [Step 1] Loading CSV Datasets ---")
    movies_df = pd.read_csv('movies.csv')
    credits_df = pd.read_csv('credits.csv')
    tv_shows_df = pd.read_csv('tv_shows.csv')
    print("Dataset 'movies.csv' loaded successfully.")
    print("Dataset 'credits.csv' loaded successfully.")
    print("Dataset 'tv_shows.csv' loaded successfully.\n")
    return movies_df, credits_df, tv_shows_df

def display_structural_properties(df, dataset_name):
    """
    Syllabus Topic: Exploratory Data Analysis (EDA) Checklist
    Systematically inspects structural properties using:
    - .shape
    - .dtypes
    - memory_usage()
    - .info()
    """
    print("=" * 60)
    print(f" STRUCTURAL PROPERTIES AUDIT: {dataset_name.upper()}")
    print("=" * 60)
    
    # A. Shape of the DataFrame
    print(f"1. Dataset Shape (Rows, Columns): {df.shape}")
    
    # B. Data Types for each column
    print("\n2. Data Types (.dtypes):")
    print(df.dtypes)
    
    # C. Memory Usage Details
    print("\n3. Memory Usage (per column in bytes):")
    mem_usage = df.memory_usage(deep=True)
    print(mem_usage)
    total_mem_mb = mem_usage.sum() / (1024 ** 2)
    print(f"--> Total Deep Memory Footprint: {total_mem_mb:.3f} MB")
    
    # D. Info Summary (prints columns, non-null counts, types)
    print("\n4. General Information (.info()):")
    df.info()
    print("=" * 60 + "\n")

def merge_movie_datasets(movies_df, credits_df):
    """
    Syllabus Topic: DataFrame Merging
    Combines movies and credits on their unique identifier.
    In movies.csv: 'movie_id' is the primary identifier.
    In credits.csv: 'id' is the primary identifier.
    Merging also maps the 'title' column which is shared by both, ensuring no duplicate title fields.
    """
    print("--- [Step 3] Merging 'movies.csv' and 'credits.csv' ---")
    # Perform inner merge on unique ID ('movie_id' matching 'id') and shared 'title'
    merged_df = pd.merge(movies_df, credits_df, left_on=['movie_id', 'title'], right_on=['id', 'title'])
    print(f"Successfully merged. Shape of combined movie dataset: {merged_df.shape}\n")
    return merged_df

def clean_movie_dataframe(df):
    """
    Syllabus Topic: Data Cleaning & Preprocessing (Movies)
    Cleans the merged movie DataFrame by:
    - Detecting and handling missing values (fillna / dropna)
    - Dropping duplicates (drop_duplicates)
    - Filtering records where essential textual fields are empty
    """
    print("--- [Step 4A] Preprocessing and Cleaning Movies Data ---")
    initial_row_count = len(df)
    
    # A. Duplicate Removal
    # Record duplicate count before dropping
    duplicates_count = df.duplicated().sum()
    df_cleaned = df.drop_duplicates().copy()
    
    # B. Missing Value Identification
    print("Missing values per column before handling:")
    missing_counts = df_cleaned.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # C. Missing Value Imputation/Handling
    # Impute non-essential optional columns
    if 'homepage' in df_cleaned.columns:
        df_cleaned['homepage'] = df_cleaned['homepage'].fillna('Not Available')
    if 'tagline' in df_cleaned.columns:
        df_cleaned['tagline'] = df_cleaned['tagline'].fillna('Not Available')
    if 'release_date' in df_cleaned.columns:
        df_cleaned['release_date'] = df_cleaned['release_date'].fillna('Unknown')
    # Impute numeric missing values (runtime) with median
    if 'runtime' in df_cleaned.columns:
        median_runtime = df_cleaned['runtime'].median()
        df_cleaned['runtime'] = df_cleaned['runtime'].fillna(median_runtime)
        
    # D. Filtering Essential Textual Fields
    # Essential columns: 'title' and 'overview' must not be null and must not be empty or whitespace-only
    # First, drop any rows where title or overview are null
    df_cleaned = df_cleaned.dropna(subset=['title', 'overview'])
    # Second, filter out rows where title or overview are empty/whitespace strings
    df_cleaned = df_cleaned[
        (df_cleaned['title'].str.strip() != '') & 
        (df_cleaned['overview'].str.strip() != '')
    ]
    
    final_row_count = len(df_cleaned)
    text_removed_count = initial_row_count - duplicates_count - final_row_count
    
    print(f"Movie cleaning finished. Remaining rows: {final_row_count}\n")
    return df_cleaned, initial_row_count, duplicates_count, text_removed_count, final_row_count

def clean_tv_dataframe(df):
    """
    Syllabus Topic: Data Cleaning & Preprocessing (TV Shows)
    Cleans the TV shows DataFrame by:
    - Detecting and handling missing values (fillna / dropna)
    - Dropping duplicates (drop_duplicates)
    - Filtering records where essential textual fields ('name', 'overview') are empty
    """
    print("--- [Step 4B] Preprocessing and Cleaning TV Shows Data ---")
    initial_row_count = len(df)
    
    # A. Duplicate Removal
    duplicates_count = df.duplicated().sum()
    df_cleaned = df.drop_duplicates().copy()
    
    # B. Missing Value Identification
    print("Missing values per column before handling:")
    missing_counts = df_cleaned.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # C. Missing Value Imputation/Handling
    # Impute non-essential path and date variables
    if 'backdrop_path' in df_cleaned.columns:
        df_cleaned['backdrop_path'] = df_cleaned['backdrop_path'].fillna('Not Available')
    if 'poster_path' in df_cleaned.columns:
        df_cleaned['poster_path'] = df_cleaned['poster_path'].fillna('Not Available')
    if 'first_air_date' in df_cleaned.columns:
        df_cleaned['first_air_date'] = df_cleaned['first_air_date'].fillna('Unknown')
        
    # D. Filtering Essential Textual Fields
    # Essential columns for TV: 'name' and 'overview' must not be null and not empty/whitespace
    df_cleaned = df_cleaned.dropna(subset=['name', 'overview'])
    df_cleaned = df_cleaned[
        (df_cleaned['name'].str.strip() != '') & 
        (df_cleaned['overview'].str.strip() != '')
    ]
    
    final_row_count = len(df_cleaned)
    text_removed_count = initial_row_count - duplicates_count - final_row_count
    
    print(f"TV Show cleaning finished. Remaining rows: {final_row_count}\n")
    return df_cleaned, initial_row_count, duplicates_count, text_removed_count, final_row_count

def standardize_tv_layout(tv_df):
    """
    Syllabus Topic: Column Layout Standardization
    Renames TV dataset 'name' column to 'title' so that it perfectly mirrors
    the structure and keys of the movie dataset, facilitating cross-dataset integration.
    """
    print("--- [Step 5] Standardizing Column Layout ---")
    # Rename column 'name' to 'title'
    tv_standardized = tv_df.rename(columns={'name': 'title'})
    print("Column 'name' renamed to 'title' in TV shows dataset.")
    print("Columns are now aligned between Movies and TV Shows datasets.\n")
    return tv_standardized

def print_cleaning_report(movies_stats, tv_stats):
    """
    Syllabus Topic: Reporting & Output Summary
    Prints a formatted, clean report of rows processed, duplicates dropped,
    text filters applied, and rows remaining.
    """
    m_init, m_dup, m_text, m_final = movies_stats
    tv_init, tv_dup, tv_text, tv_final = tv_stats
    
    print("=" * 70)
    print("               CINE MATCH DATA PREPROCESSING SUMMARY REPORT")
    print("=" * 70)
    print(f"{'Processing Stage / Metric':<38} | {'Movies':<12} | {'TV Shows':<12}")
    print("-" * 70)
    print(f"{'1. Raw / Merged Initial Rows':<38} | {m_init:<12} | {tv_init:<12}")
    print(f"{'2. Duplicates Removed':<38} | {m_dup:<12} | {tv_dup:<12}")
    print(f"{'3. Empty/Null Text Rows Filtered':<38} | {m_text:<12} | {tv_text:<12}")
    print(f"{'4. Final Cleaned Rows Remaining':<38} | {m_final:<12} | {tv_final:<12}")
    print("-" * 70)
    print(f"{'Dataset Survival Rate (%)':<38} | {m_final/m_init*100:11.2f}% | {tv_final/tv_init*100:11.2f}%")
    print("=" * 70)
    print("Flight path activated: All data points clean and ready for Phase 2!")
    print("=" * 70 + "\n")

def main():
    print("======================================================================")
    print(" CineMatch Phase 1, Sub-Phase 1.1: Academic EDA & Clean Flight Path")
    print("======================================================================\n")
    
    # Step 1: Load CSV files
    movies, credits, tv_shows = load_datasets()
    
    # Step 2: Display structural properties for each dataset
    display_structural_properties(movies, "movies.csv")
    display_structural_properties(credits, "credits.csv")
    display_structural_properties(tv_shows, "tv_shows.csv")
    
    # Step 3: Merge movies and credits datasets
    merged_movies = merge_movie_datasets(movies, credits)
    
    # Step 4: Clean dataframes independently
    cleaned_movies, m_init, m_dup, m_text, m_final = clean_movie_dataframe(merged_movies)
    cleaned_tv, tv_init, tv_dup, tv_text, tv_final = clean_tv_dataframe(tv_shows)
    
    # Step 5: Standardize the TV show layout (rename name to title)
    cleaned_tv = standardize_tv_layout(cleaned_tv)
    
    # Step 6: Print clean data-cleaning summary report
    print_cleaning_report(
        (m_init, m_dup, m_text, m_final),
        (tv_init, tv_dup, tv_text, tv_final)
    )

if __name__ == '__main__':
    main()
