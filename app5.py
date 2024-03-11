import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("Hotel Reviews Sentiment Analysis")

# Text area for user input
user_input = st.text_area("Enter your hotel review:")

if user_input:
    # Analyze sentiment
    sentiment = sia.polarity_scores(user_input)
    compound_score = sentiment['compound']

    # Determine sentiment label
    if compound_score >= 0.05:
        sentiment_label = 'Positive'
    elif compound_score <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    # Display sentiment analysis results
    st.write(f"Sentiment: {sentiment_label} (Compound Score: {compound_score:.2f})")

