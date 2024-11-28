from textblob import TextBlob
import pandas as pd

# Analyze sentiment of a single text
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)
    return sentiment_score

# Apply sentiment analysis to the dataframe
def apply_sentiment_analysis(df):
    df['sentiment_score'] = df['title'].apply(analyze_sentiment)
    df['sentiment_score'] = df['sentiment_score'].fillna(0)  # Replace NaN with 0
    return df

# Save sentiment scores
def save_sentiment_scores(df, filename='data/cleaned_data_with_sentiment.csv'):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_data.csv')
    df = apply_sentiment_analysis(df)
    save_sentiment_scores(df)
    print("Sentiment analysis complete.")
