import json
import pandas as pd

# Load data from JSON
def load_data(filename='data/raw_data.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Preprocess the data (remove unnecessary columns, handle missing values, etc.)
def preprocess_data(data):
    df = pd.DataFrame(data)
    # Remove any rows with missing values
    df = df.dropna()
    # You can add more preprocessing steps here if needed
    return df

# Save cleaned data
def save_cleaned_data(df, filename='data/cleaned_data.csv'):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    data = load_data('data/raw_data.json')
    df = preprocess_data(data)
    save_cleaned_data(df)
    print("Data Preprocessed and Saved.")
