import os
import time
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from dotenv import load_dotenv  # Import dotenv to load environment variables
import praw

# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize NLTK Sentiment Analyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Global Configurations
pd.set_option("display.max_colwidth", 1000)
CHROMEDRIVER_PATH = os.path.abspath("./chromedriver-win64/chromedriver.exe")
CSV_FILE_PATH = os.path.abspath("./yahoo_news_data.csv")

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)


class StockService:
    """
    Service class to fetch stock data from various sources.
    """

    @staticmethod
    def fetch_stock_data(symbol, company=""):
        """
        Fetch stock data from multiple sources for a given stock symbol.
        """
        try:
            news_articles = analyze_combined_news(symbol)
            return news_articles
        except requests.exceptions.RequestException as e:
            print(f"Error fetching stock data: {e}")
            return {"error": str(e)}


# Sentiment Analysis Function
def analyze_sentiment(text):
    """
    Analyze sentiment and return a score rounded to two decimal places.
    """
    sentiment_score = sia.polarity_scores(text)["compound"]
    return round(sentiment_score, 2)


def fetch_yahoo_finance_news(symbol):
    """
    Fetch the latest 20 Yahoo Finance news articles for a given symbol from the CSV file.
    """
    try:
        # Check if the CSV file exists
        if not os.path.exists(CSV_FILE_PATH):
            print("Yahoo Finance news data not found. Please run the scraper first.")
            return pd.DataFrame(
                columns=["Title", "URL", "Date", "Summary", "Sentiment"]
            )

        # Load the data from the CSV file
        df = pd.read_csv(CSV_FILE_PATH)

        # Filter by symbol
        symbol_news = df[df["Symbol"] == symbol]

        # If no news found for the symbol, return an empty DataFrame
        if symbol_news.empty:
            print(f"No news found for symbol: {symbol}")
            return pd.DataFrame(
                columns=["Title", "URL", "Date", "Summary", "Sentiment"]
            )

        # Select the first 20 rows and required columns
        latest_news = symbol_news.tail(20)[
            ["Title", "URL", "Date", "Summary", "Sentiment"]
        ]

        return latest_news.reset_index(drop=True)

    except Exception as e:
        print(f"Error fetching Yahoo Finance news from CSV for '{symbol}': {e}")
        return pd.DataFrame(columns=["Title", "URL", "Date", "Summary", "Sentiment"])


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


def fetch_articles_with_sentiments(keyword, start_date, sources=None):
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
            q=keyword,
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
            datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%d-%b-%Y")
            if date != "No date available"
            else date
        )
        articles_data.append(
            (article["title"], article["url"], date, article["description"], sentiment)
        )

    # Return as DataFrame
    return pd.DataFrame(
        articles_data, columns=["Title", "URL", "Date", "Summary", "Sentiment"]
    )


def fetch_reddit_posts(symbol):
    """
    Fetch Reddit posts from the 'stocks' subreddit related to the given symbol.
    Analyze the sentiment of each post and return the results as a DataFrame.
    """
    try:
        # Search for posts in the 'stocks' subreddit
        reddit_posts = reddit.subreddit("stocks").search(symbol, limit=50)
        articles_data = []

        for post in reddit_posts:
            # Analyze the sentiment of the post title
            summary = (
                post.selftext[:100]
                if hasattr(post, "selftext")
                else "No summary available"
            )
            sentiment_score = analyze_sentiment(f"{post.title} {summary}")

            # Append processed data to the list
            articles_data.append(
                {
                    "Title": post.title,
                    "URL": f"https://reddit.com{post.permalink}",
                    "Date": datetime.fromtimestamp(post.created_utc).strftime(
                        "%d-%b-%Y"
                    ),
                    "Summary": summary,
                    "Sentiment": sentiment_score,
                }
            )

        # Return the data as a DataFrame
        return pd.DataFrame(
            articles_data, columns=["Title", "URL", "Date", "Summary", "Sentiment"]
        )
    except Exception as e:
        print(f"Error fetching Reddit posts for '{symbol}': {e}")
        # Return an empty DataFrame in case of an error
        return pd.DataFrame(columns=["Title", "URL", "Date", "Summary", "Sentiment"])


# Combine News from Multiple Sources
def analyze_combined_news(symbol):
    """
    Combine Yahoo Finance and NewsAPI articles for the given stock symbol.
    """
    try:
        yahoo_news = fetch_yahoo_finance_news(symbol)
    except Exception as e:
        print(f"Error fetching Yahoo Finance news for '{symbol}': {e}")
        yahoo_news = pd.DataFrame(
            columns=["Title", "URL", "Date", "Summary", "Sentiment"]
        )

    try:
        newsapi_articles = fetch_newsapi_articles(symbol)
    except Exception as e:
        print(f"Error fetching NewsAPI articles for '{symbol}': {e}")
        newsapi_articles = pd.DataFrame(
            columns=["Title", "URL", "Date", "Summary", "Sentiment"]
        )

    try:
        reddit_posts = fetch_reddit_posts(symbol)
    except Exception as e:
        print(f"Error fetching Reddit articles for '{symbol}': {e}")
        reddit_posts = pd.DataFrame(
            columns=["Title", "URL", "Date", "Summary", "Sentiment"]
        )

    # Combine the results
    combined_news = pd.concat(
        [yahoo_news, newsapi_articles, reddit_posts], ignore_index=True
    )
    print(combined_news)

    return combined_news.to_dict(orient="records")
