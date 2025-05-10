from textblob import TextBlob

def analyze_sentiment(text):
    if not text: return 0
    return TextBlob(text).sentiment.polarity

def apply_sentiment(df):
    df['Sentiment'] = df['Cleaned_Notes'].apply(analyze_sentiment)
    return df