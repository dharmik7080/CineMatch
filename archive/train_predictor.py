"""
CineMatch Project - Phase 4: Machine Learning & Predictive Modeling
Sub-Phase 4.3: Multiple Linear Regression Model Training

Syllabus Reference: Unit 4: Regression Analysis & Predictive Modeling
- Multiple Linear Regression (MLR) formulation
- Coefficient weights (Beta coefficients)
- Model training and evaluation (R2 score, Mean Absolute Error)
- Model serialization (Pickle)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Import dataset loader and pre-cleaning modules to keep it clean and compliant
from eda_and_cleaning import load_datasets, merge_movie_datasets, clean_movie_dataframe

def main():
    print("======================================================================")
    print(" CineMatch Phase 4: Multiple Linear Regression Model Training")
    print("======================================================================\n")
    
    # Step 1: Load and prepare the processed movies dataset
    print("--- [Step 1] Loading and Pre-cleaning Movies Dataset ---")
    movies_raw, credits_raw, _ = load_datasets()
    merged_movies = merge_movie_datasets(movies_raw, credits_raw)
    
    # We clean the dataset using our previously defined pipeline
    cleaned_df, _, _, _, _ = clean_movie_dataframe(merged_movies)
    print(f"Initial processed dataset shape: {cleaned_df.shape}\n")
    
    # Step 2 & 3: Clean and isolate features
    # Syllabus Topic: Feature Isolation & Purity Check (Unit 4)
    # We select:
    #   - Independent variables (X): 'budget' (x1) and 'popularity' (x2)
    #   - Dependent target (y): 'vote_average' (y)
    # Dropping records where budget, popularity, or vote_average are zero or missing.
    print("--- [Steps 2 & 3] Isolating Features and Cleaning Zero/Null Values ---")
    
    # Create copy to prevent warnings
    reg_df = cleaned_df[['title', 'budget', 'popularity', 'vote_average']].copy()
    
    # Filter: Drop rows containing missing values or zeros in these specific coordinates
    # Fiscally, a movie budget of 0 is mathematically invalid for financial forecasting,
    # and popularity/rating values must be active.
    reg_df = reg_df[
        (reg_df['budget'] > 0) & 
        (reg_df['popularity'] > 0) & 
        (reg_df['vote_average'] > 0)
    ]
    
    print(f"Purity filter complete. Model training data shape: {reg_df.shape}")
    
    X = reg_df[['budget', 'popularity']]
    y = reg_df['vote_average']
    
    # Step 4: Split dataset into 80% train and 20% test arrays
    # Syllabus Topic: Model Selection & Train-Test Split (Unit 4)
    print("\n--- [Step 4] Splitting Dataset (80% Train, 20% Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set dimension: {X_train.shape}")
    print(f"Testing set dimension:  {X_test.shape}")
    
    # Step 5: Fit and train the Multiple Linear Regression model
    # Syllabus Topic: Multiple Linear Regression Fitting (Unit 4)
    # Mathematical Equation: y_hat = beta_0 + beta_1 * x_1 + beta_2 * x_2
    # Where:
    #   - y_hat: predicted movie rating (vote_average)
    #   - x_1: budget
    #   - x_2: popularity
    #   - beta_0: intercept
    #   - beta_1, beta_2: feature weights / regression coefficients
    print("\n--- [Step 5] Training Multiple Linear Regression Model ---")
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("Model fitting successfully completed.")
    
    # Step 6: Evaluate Model Performance & calculate weights
    # Syllabus Topic: Metric Evaluation & Parameter Estimation (Unit 4)
    print("\n--- [Step 6] Evaluating Model Parameters and Performance ---")
    
    # Perform prediction on test set
    y_pred = regressor.predict(X_test)
    
    # Calculate evaluation metrics
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Extract coefficients and intercept
    intercept = regressor.intercept_
    coef_budget = regressor.coef_[0]
    coef_popularity = regressor.coef_[1]
    
    # Output metrics to terminal
    print("=" * 60)
    print("                 REGRESSION MODEL METRICS REPORT")
    print("=" * 60)
    print(f"Model Intercept (beta_0):                 {intercept:.6f}")
    print(f"Budget Coefficient Weight (beta_1):       {coef_budget:.12e}")
    print(f"Popularity Coefficient Weight (beta_2):   {coef_popularity:.6f}")
    print("-" * 60)
    print(f"Mean Absolute Error (MAE):                {mae:.4f} rating points")
    print(f"Mean Squared Error (MSE):                 {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):           {rmse:.4f}")
    print(f"Coefficient of Determination (R² Score):   {r2_score:.4f}")
    print("=" * 60)
    
    print("\nRegression Equation:")
    print(f"  vote_average = {intercept:.4f} + ({coef_budget:.4e} * budget) + ({coef_popularity:.4f} * popularity)")
    
    # Step 7: Serialize and save the trained model object using pickle
    print("\n--- [Step 7] Serializing and Saving Trained Model ---")
    model_output_path = 'success_predictor.pkl'
    with open(model_output_path, 'wb') as f:
        pickle.dump(regressor, f)
    print(f"Successfully saved trained model object to '{model_output_path}'.\n")
    print("Flight path activated: Predictive analytics asset successfully trained!")
    print("======================================================================\n")

if __name__ == '__main__':
    main()
