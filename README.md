# Stock Prediction Using Sentiment Analysis

## Overview
This project leverages natural language processing (NLP) and machine learning to predict stock market movements based on the sentiment of financial news articles. By scraping financial news data, performing sentiment analysis, and using these insights as features for stock movement prediction, this project provides a comprehensive pipeline for financial forecasting.

### Key Features
- **Automated Data Scraping**: Gathers news articles from financial news websites for real-time and historical data analysis.
- **Sentiment Analysis**: Analyzes news headlines and articles to determine their sentiment (positive, negative, or neutral).
- **Stock Movement Prediction**: Uses machine learning algorithms to predict stock price trends based on sentiment scores and other derived features.
- **Modular Design**: Clear separation of data collection, preprocessing, model training, and evaluation steps for ease of use and extensibility.

---

## Project Objectives
1. **Demonstrate the use of sentiment analysis for financial decision-making.**
2. **Build a reliable prediction model using machine learning techniques.**
3. **Provide a scalable and modular framework for future extensions.**

---

## Prerequisites
### Software Requirements
- **Python**: Version 3.8 or above.
- **Jupyter Notebook**: For visualization and experimentation.
- **Git**: Version control system.

### Libraries and Dependencies
The following Python libraries are required to run this project:
- `pandas` (Data manipulation)
- `numpy` (Numerical computations)
- `scikit-learn` (Machine learning)
- `nltk` (Natural language processing)
- `beautifulsoup4` (Web scraping)
- `matplotlib` (Data visualization)
- `seaborn` (Advanced visualization)

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Setup Instructions

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/<DikshaKapse>/Stock-Prediction-Sentiment-Analysis.git
cd Stock-Prediction-Sentiment-Analysis
```

### Step 2: Set Up the Environment
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Run the Scraper
The scraper fetches financial news data and stores it in the `data/` folder. Run it using:
```bash
python data_scraping.py
```
### step 4 :Data Preprocessing
It processes and cleans the data
Run it using:
```bash
python data_preprocessing.py
```
### step 5:Sentiment Analysis:
Run it using:
```bash
python sentiment_analysis.py
```
### step 6:Feature Extraction
It extracts the features from data
Run it using:
```bash
python feature_extractin.py
```

### Step 7: Train the Model
Train the prediction model using the preprocessed data:
```bash
python model_training.py
```

### Step 8: Evaluate the Model
Evaluate the model's performance and visualize results:
```bash
python model_evaluation.py
```

---


## Implementation Details

### 1. Data Scraping
- **Source**: Financial news websites (e.g., Reuters, Bloomberg).
- **Tools**: `BeautifulSoup` for HTML parsing.
- **Challenges**: Handling inconsistent website structures, rate limiting, and anti-scraping mechanisms.
- **Solutions**:
  - Rotating user agents.
  - Adding delays to requests.
  - Using an IP rotation service (if necessary).

### 2. Sentiment Analysis
- **Techniques**:
  - Preprocessing text: Removing stop words, punctuation, and irrelevant tokens.
  - Tokenization using `nltk`.
  - Sentiment scoring with a pretrained sentiment analysis model.
- **Challenges**:
  - Ambiguity in language.
  - Context understanding in financial news.

### 3. Stock Movement Prediction
- **Features**:
  - Sentiment scores (positive, negative, neutral).
  - Historical stock prices.
  - Moving averages and technical indicators.
- **Algorithms**: Random Forest, Support Vector Machine (SVM), and Gradient Boosting.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 4. Model Evaluation
- Confusion Matrix and classification reports provide detailed insights into model performance.

---

## Results and Insights
- The sentiment of news articles shows a strong correlation with stock price movements.
- Incorporating additional features like trading volume and global indices could further improve accuracy.

---

## Challenges and Solutions
### Challenges
- Dealing with noisy data and incomplete news articles.
- Overfitting on small datasets.

### Solutions
- Extensive preprocessing and feature selection.
- Regularization techniques and hyperparameter tuning.

---
