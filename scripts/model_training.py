import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import scripts.feature_extraction
import importlib
importlib.reload(scripts.feature_extraction)
import sys
import os

# Add the 'scripts' folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Train a model to predict stock sentiment based on Reddit data
def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    return predictions

# Save the trained model
def save_model(model, filename='models/stock_prediction_model.pkl'):
    joblib.dump(model, filename)

# Save the vectorizer
def save_vectorizer(vectorizer, filename='models/vectorizer.pkl'):
    joblib.dump(vectorizer, filename)

if __name__ == "__main__":
    from scripts.feature_extraction import extract_features, split_data

    df = pd.read_csv('data/cleaned_data_with_sentiment.csv')
    X, y, vectorizer = extract_features(df)  # Extract features and get the vectorizer
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Save both model and vectorizer
    save_model(model)
    save_vectorizer(vectorizer)  # Save the vectorizer
    print("Model training complete, model and vectorizer saved.")
