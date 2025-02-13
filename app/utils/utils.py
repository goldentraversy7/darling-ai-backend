from transformers import pipeline
import os

finbert_sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def analyze_sentiment(text):
    """
    Analyze sentiment and return a numerical sentiment score.
    """
    result = finbert_sentiment(text)
    sentiment_label = result[0]["label"]
    confidence_score = result[0]["score"]

    # Convert sentiment label to numerical value
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    sentiment_value = mapping[sentiment_label]

    # Compute final sentiment score
    sentiment_score = sentiment_value * confidence_score

    return round(sentiment_score, 4)  # Rounded for consistency
