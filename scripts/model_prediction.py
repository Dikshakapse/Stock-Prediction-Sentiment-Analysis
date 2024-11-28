import joblib
import pandas as pd
from scripts.feature_extraction import extract_features
import os

# Correct path using the absolute path
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')
vectorizer = joblib.load(vectorizer_path)


# Load the saved model and vectorizer
model = joblib.load('models/stock_prediction_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')  # Load the saved vectorizer

# New data (make sure it's processed and features are extracted as before)
new_data = pd.DataFrame({
    'title': ['New Reddit post example']  # Example data
})

# Extract features using the saved vectorizer (apply the same transformation)
X_new = vectorizer.transform(new_data['title'])  # Use the existing vectorizer for transformation

# Make predictions
predictions = model.predict(X_new)
print(f"Predictions: {predictions}")
