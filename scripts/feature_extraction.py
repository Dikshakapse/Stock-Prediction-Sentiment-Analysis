from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Extract features from the dataset (using TF-IDF vectorizer)

def extract_features(df, include_sentiment=True):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['title'])  # Using 'title' column for feature extraction

    if include_sentiment:
        y = df['sentiment_score']  # Use sentiment_score if available
        return X, y, vectorizer
    else:
        return X, vectorizer  # Return X and vectorizer for new data without sentiment_score
    # scripts/feature_extraction.py


    



# Split data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_data_with_sentiment.csv')
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training data: {len(X_train)} samples, Test data: {len(X_test)} samples.")
