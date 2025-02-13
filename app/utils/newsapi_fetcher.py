import os
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
import numpy as np
from utils import analyze_sentiment


# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Global Configurations
pd.set_option("display.max_colwidth", 1000)
CSV_FILE_PATH = os.path.abspath("./yahoo_news_data.csv")


# NewsAPI Integration
def fetch_newsapi_articles(symbol):
    """
    Fetch articles from NewsAPI for a given symbol and date range.
    """
    sources = fetch_newsapi_sources(category="business")
    today = datetime.today().strftime("%d-%b-%Y")
    return fetch_articles_with_sentiments(symbol, today, sources)


def fetch_newsapi_sources(category=None):
    """
    Fetch available news sources from NewsAPI filtered by category.
    """
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    sources_data = newsapi.get_sources()

    if category:
        sources = [
            source["id"]
            for source in sources_data["sources"]
            if source["category"] == category and source["language"] == "en"
        ]
    else:
        sources = [
            source["id"]
            for source in sources_data["sources"]
            if source["language"] == "en"
        ]

    return sources


def fetch_articles_with_sentiments(symbol, start_date, sources=None):
    """
    Fetch articles from NewsAPI within a 7-day date range and analyze sentiments.
    """
    # Initialize NewsAPI
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # Convert start_date to datetime and calculate date range
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%d-%b-%Y")
    from_date = start_date - timedelta(days=7)

    try:
        # Fetch articles
        articles = newsapi.get_everything(
            q=symbol,
            from_param=from_date.isoformat(),
            to=start_date.isoformat(),
            language="en",
            sources=",".join(sources) if sources else None,
            sort_by="relevancy",
            page_size=100,
        )
    except Exception as e:
        print(f"Error fetching articles from NewsAPI: {e}")
        return pd.DataFrame(columns=["Title", "URL", "Date", "Summary", "Sentiment"])

    # Process articles
    seen_titles = set()
    articles_data = []

    for article in articles.get("articles", []):
        if article["title"] in seen_titles:
            continue
        seen_titles.add(article["title"])
        content = f"{article['title']}. {article['description']}"
        sentiment = analyze_sentiment(content)
        date = article.get("publishedAt", "No date available")
        date = (
            datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
            if date != "No date available"
            else date
        )
        articles_data.append(
            (article["title"], article["url"], date, article["description"], sentiment)
        )

    print(
        pd.DataFrame(
            articles_data, columns=["Title", "URL", "Date", "Summary", "Sentiment"]
        )
    )

    # Return as DataFrame
    return pd.DataFrame(
        articles_data, columns=["Title", "URL", "Date", "Summary", "Sentiment"]
    )


# Example usage for testing
if __name__ == "__main__":
    fetch_newsapi_articles("AAPL")
