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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from dotenv import load_dotenv  # Import dotenv to load environment variables
import praw
from io import BytesIO

matplotlib.use("Agg")

# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

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
            # Fetch news articles (ensure `analyze_combined_news` is defined elsewhere)
            news_articles = analyze_combined_news(symbol)

            # Analyze stock data and generate the plot
            stock_data, plot_buffer, plot_figure = analyze_stock_data(symbol)

            # Handle NaN values and convert DataFrame to JSON-serializable format
            if stock_data is not None:
                # Replace NaN with None (which converts to null in JSON)
                stock_data = stock_data.replace(
                    [np.nan, np.inf, -np.inf], None
                ).to_dict(orient="records")
            else:
                stock_data = []

            # Encode plot as Base64
            plot_base64 = (
                StockService.encode_plot_as_base64(plot_figure) if plot_figure else None
            )

            # Create response dictionary
            response = {
                "stock_data": stock_data,
                "plot": plot_base64,
                "news_articles": news_articles,
            }

            return response  # Convert response to JSON
        except Exception as e:
            print(f"Unexpected error: {e}")
            return json.dumps({"error": f"Unexpected error: {str(e)}"})

    @staticmethod
    def encode_plot_as_base64(plot_figure):
        """
        Encode a matplotlib plot as a Base64 string.
        """
        try:
            # Create a BytesIO buffer
            buffer = BytesIO()
            plot_figure.savefig(buffer, format="png")  # Save the plot to the buffer
            buffer.seek(0)  # Move to the start of the buffer
            # Encode the buffer to Base64
            plot_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()  # Close the buffer to release resources
            return f"data:image/png;base64,{plot_base64}"
        except Exception as e:
            print(f"Error encoding plot as Base64: {e}")
            raise


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


def fetch_reddit_posts(symbol):
    """
    Fetch Reddit posts from the 'stocks' subreddit related to the given symbol.
    Analyze the sentiment of each post and return the results as a DataFrame.
    """
    try:
        # Search for posts in the 'stocks' subreddit
        reddit_posts = reddit.subreddit("stocks").search(symbol, limit=20)
        articles_data = []

        for post in reddit_posts:
            # Analyze the sentiment of the post title
            summary = (
                f"{post.selftext[:100]}..."
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
                        "%Y-%m-%d"
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

    return combined_news.to_dict(orient="records")


def analyze_stock_data(symbol):
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(symbol)

        # Perform analysis
        stock_data = calculate_rsi(stock_data)
        stock_data = detect_market_trends(stock_data)

        # Generate the plot
        plot_buffer, plot_figure = plot_market_trends(stock_data, symbol)
        stock_data = stock_data[::-1]  # Reverse the DataFrame
        stock_data["dTime"] = stock_data.index
        print(stock_data)
        # Return the analyzed data and plot
        return stock_data, plot_buffer, plot_figure

    except Exception as e:
        # Log the error for debugging
        print(f"Error in analyzing stock data for symbol '{symbol}': {e}")

        # Return empty values or default placeholders
        return None, None, None


# Function to fetch stock data with 15-minute interval
def fetch_stock_data(symbol, interval="15min"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    timeseries = data.get(f"Time Series ({interval})", {})
    if not timeseries:
        raise ValueError("Failed to fetch data. Check the symbol or API usage limits.")

    df = pd.DataFrame.from_dict(timeseries, orient="index")
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    ).astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# Function to calculate RSI
def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# Detect bull and bear trends based on moving averages
def detect_market_trends(df, short_window=20, long_window=50):
    df["SMA_short"] = df["close"].rolling(window=short_window).mean()
    df["SMA_long"] = df["close"].rolling(window=long_window).mean()
    df["Trend"] = np.where(df["SMA_short"] > df["SMA_long"], "Bull", "Bear")
    return df


def plot_market_trends(stock_data, symbol):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the data
    ax.plot(
        stock_data.index,
        stock_data["close"],
        label="Close Price",
        color="blue",
        linewidth=1,
    )
    ax.plot(
        stock_data.index,
        stock_data["SMA_short"],
        label="Short-term SMA (20)",
        color="orange",
        linewidth=1,
    )
    ax.plot(
        stock_data.index,
        stock_data["SMA_long"],
        label="Long-term SMA (50)",
        color="green",
        linewidth=1,
    )

    # Highlight bull and bear markets
    ax.fill_between(
        stock_data.index,
        stock_data["close"],
        stock_data["SMA_long"],
        where=(stock_data["close"] > stock_data["SMA_long"]),
        color="green",
        alpha=0.2,
        label="Bull Market",
    )
    ax.fill_between(
        stock_data.index,
        stock_data["close"],
        stock_data["SMA_long"],
        where=(stock_data["close"] < stock_data["SMA_long"]),
        color="red",
        alpha=0.2,
        label="Bear Market",
    )

    # Customize the plot
    ax.set_title(f"{symbol.upper()} Market Trends (15-Minute Interval)", fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    buffer.seek(0)
    plt.close()

    return buffer, fig
