# scripts/model_evaluation.py
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scripts.feature_extraction import extract_features, split_data

# Load the trained model
model = joblib.load('models/stock_prediction_model.pkl')

# Load the data for evaluation
df = pd.read_csv('data/cleaned_data_with_sentiment.csv')

# Extract features and split data
X, y, vectorizer = extract_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Evaluate the model's performance
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    print("First 10 Predictions:", predictions[:10])
    return predictions

evaluate_model(model, X_test, y_test)
print(f"X_test shape: {X_test.shape}")
print(f"True values (y_test): {y_test}")
print(X_train.toarray()[:5])
print(vectorizer.get_feature_names_out()[:10])
#print(f"Predictions: {predictions}")
print(f"True values: {y_test}")


    

