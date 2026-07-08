import os
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags (like <br /><br />)
    text = re.sub(r'<[^>]*>', ' ', text)
    # Lowercase the text
    text = text.lower()
    # Keep only alphabet characters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    dataset_path = "sentiment_training_data.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    print("Preprocessing text...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Map sentiment column to binary integers
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Drop rows with NaN labels or reviews if any
    df = df.dropna(subset=['cleaned_review', 'label'])

    print("Splitting into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], 
        df['label'], 
        test_size=0.2, 
        random_state=42
    )

    print("Vectorizing text with TF-IDF...")
    # Use english stop words to convert text into numerical vectors
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}\n")

    # Export
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("models", "sentiment_model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    print("Training process completed successfully.")

if __name__ == "__main__":
    main()
