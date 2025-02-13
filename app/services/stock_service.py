import os
import time
import json
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import pipeline
from newsapi import NewsApiClient
from dotenv import load_dotenv
import yfinance as yf
import numpy as np

matplotlib.use("Agg")

# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Global Configurations
pd.set_option("display.max_colwidth", 1000)
CSV_FILE_PATH = os.path.abspath("./yahoo_news_data.csv")


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
            # Fetch news articles (ensure `analyze_combined_news` is defined elsewhere)
            # news_articles = analyze_combined_news(symbol)

            # Analyze stock data and generate the plot
            stock_data = StockService.analyze_stock_data(symbol)

            # Create response dictionary
            response = {
                "stock_data": stock_data,
                # "news_articles": news_articles,
            }

            return response  # Convert response to JSON
        except Exception as e:
            print(f"Unexpected error: {e}")
            return json.dumps({"error": f"Unexpected error: {str(e)}"})

    @staticmethod
    def analyze_stock_data(symbol):
        try:
            # Fetch stock data
            stock_data = fetch_stock_data(symbol)

            # Generate the plot
            stock_data["dTime"] = stock_data.index
            # Return the analyzed data and plot
            # Handle NaN values and convert DataFrame to JSON-serializable format
            if stock_data is not None:
                # Replace NaN with None (which converts to null in JSON)
                stock_data = stock_data.replace(
                    [np.nan, np.inf, -np.inf], None
                ).to_dict(orient="records")
            else:
                stock_data = []

            return stock_data

        except Exception as e:
            # Log the error for debugging
            print(f"Error in analyzing stock data for symbol '{symbol}': {e}")

            # Return empty values or default placeholders
            return None


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
            datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
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

    # Combine the results
    combined_news = pd.concat([yahoo_news, newsapi_articles], ignore_index=True)

    return combined_news.to_dict(orient="records")


def calculate_technical_indicators(df):
    """
    Calculates additional technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands.
    """
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD Calculation
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Upper"] = df["SMA_20"] + (df["close"].rolling(window=20).std() * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["close"].rolling(window=20).std() * 2)

    return df


def fetch_stock_data(symbol, interval="1d", period="6mo"):
    """
    Fetch historical stock data using Yahoo Finance (yfinance).

    :param symbol: Stock ticker symbol.
    :param interval: Interval of stock data (e.g., "1d", "1wk", "1mo").
    :param period: How much history to fetch (e.g., "6mo", "1y", "5y").
    :return: Pandas DataFrame with stock data and indicators.
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return None

        # Keep relevant columns and rename
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Add stock symbol as a feature
        df["symbol"] = symbol

        # Calculate additional technical indicators
        df = calculate_technical_indicators(df)

        # Convert datetime index to string for JSON response
        df.index = df.index.strftime("%Y-%m-%d")

        return df

    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None
