import os
import time
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv  # Import dotenv to load environment variables
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import feedparser
import nltk

nltk.download("punkt")
# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
BING_API_KEY = os.getenv("BING_SEARCH_API_KEY")

# Global Configurations
pd.set_option("display.max_colwidth", 1000)
CSV_FILE_PATH = os.path.abspath("./yahoo_news_data.csv")


class BackgroundService:
    """
    Service class to fetch stock data from various sources.
    """

    @staticmethod
    def fetch_background_news(query, start_date=None, end_date=None):
        """
        Fetch stock data from multiple sources for a given stock query.
        """
        try:
            # Set today's date
            today = datetime.today().date()
            # If end_date is None or after today's date, set it to today's date
            if end_date is None or end_date > today:
                end_date = today

            # If start_date is None or after today's date, set it to 7 days before today
            if start_date is None or start_date > today:
                start_date = today - timedelta(days=7)

            # Fetch news articles (ensure `analyze_combined_news` is defined elsewhere)
            google_news = fetch_google_news(query, start_date, end_date)
            bing_news = fetch_bing_news(query, start_date, end_date)
            yahoo_news = fetch_yahoo_news(query, start_date, end_date)
            bloomberg_news = fetch_bloomberg_news(query, start_date, end_date)

            # Create response dictionary
            response = {
                "google_news": google_news,
                "bing_news": bing_news,
                "yahoo_news": yahoo_news,
                "bloomberg_news": bloomberg_news,
            }

            return response  # Convert response to JSON
        except Exception as e:
            print(f"Unexpected error: {e}")
            return json.dumps({"error": f"Unexpected error: {str(e)}"})


def fetch_google_news(query, start_date, end_date):
    try:
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        config = Config()
        config.browser_user_agent = user_agent
        googlenews = GoogleNews(start=start_date, end=end_date)
        googlenews.search(query)
        result = googlenews.result()
        df = pd.DataFrame(result)

        for i in range(2, 5):
            googlenews.getpage(i)
            result = googlenews.result()
            df = pd.concat(
                [df, pd.DataFrame(result)], ignore_index=True
            )  # Append new data to existing DataFrame

        news_list = []
        for ind in df.index:
            # article = Article(df["link"][ind], config=config)
            # article.download()
            # article.parse()
            # article.nlp()

            news_item = {
                "Title": df["title"][ind] if "title" in df else None,
                "URL": df["link"][ind] if "link" in df else None,
                "Date": (
                    pd.to_datetime(df["datetime"][ind]).strftime("%Y-%m-%d %H:%M")
                    if "datetime" in df and pd.notna(df["datetime"][ind])
                    else None
                ),
                "Desc": df["desc"][ind] if "desc" in df else None,
                "Media": df["media"][ind] if "media" in df else None,
                "Img": df["img"][ind] if "media" in df else None,
                # "Title": article.title,
                # "Article": article.text,
                # "Summary": article.summary,
            }
            news_list.append(news_item)

        news_df = pd.DataFrame(news_list)

    except Exception as e:
        print(f"Error fetching Google news for '{query}': {e}")
        news_df = pd.DataFrame(columns=["Title", "URL", "Date", "Desc", "Media", "Img"])

    return news_df.to_dict(orient="records")


def fetch_bing_news(query, start_date=None, end_date=None, count=50):
    BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"
    try:
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}

        params = {
            "q": query,
            "count": count,  # Number of results
            "mkt": "en-US",  # Market (Change as needed)
            "sortBy": "Date",  # Sort by latest news
        }

        response = requests.get(BING_ENDPOINT, headers=headers, params=params)
        data = response.json()

        if "value" not in data:
            print("No Bing news found.")
            return []

        news_list = []
        for item in data["value"]:
            news_item = {
                "Title": item["name"],
                "URL": item["url"],
                "Date": datetime.strptime(
                    item["datePublished"], "%Y-%m-%dT%H:%M:%SZ"
                ).strftime("%Y-%m-%d %H:%M"),
                "Summary": item.get("description", ""),
                "Media": (
                    item["provider"][0]["name"]
                    if "provider" in item and item["provider"]
                    else "Unknown"
                ),
                "Img": (
                    item["image"]["thumbnail"]["contentUrl"]
                    if "image" in item and "thumbnail" in item["image"]
                    else None
                ),
            }
            news_list.append(news_item)

        news_df = pd.DataFrame(news_list)

    except Exception as e:
        print(f"Error fetching Google news for '{query}': {e}")
        news_df = pd.DataFrame(columns=["Title", "URL", "Date", "Desc", "Media", "Img"])

    return news_df.to_dict(orient="records")


def fetch_yahoo_news(query):
    try:
        rss_url = f"https://news.search.yahoo.com/rss?p={query}"
        feed = feedparser.parse(rss_url)

        news_list = []
        for entry in feed.entries:
            news_item = {
                "Title": entry.title,
                "URL": entry.link,
                "Date": datetime(*entry.published_parsed[:6]).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Summary": entry.summary if "summary" in entry else "",
                "Media": "Yahoo News",
                "Image": (
                    entry.media_content[0]["url"] if "media_content" in entry else None
                ),  # Image if available
            }
            news_list.append(news_item)

        news_df = pd.DataFrame(news_list)

    except Exception as e:
        print(f"Error fetching Google news for '{query}': {e}")
        news_df = pd.DataFrame(columns=["Title", "URL", "Date", "Desc", "Media", "Img"])

    return news_df.to_dict(orient="records")


def fetch_bloomberg_news(start_date=None, end_date=None):
    try:
        # Bloomberg RSS Feed URL for Technology News (Change as needed)
        rss_url = "https://www.bloomberg.com/feed/podcast/technology.xml"
        feed = feedparser.parse(rss_url)

        news_list = []
        for entry in feed.entries:
            # Convert date to datetime object
            article_date = datetime(*entry.published_parsed[:6])

            # Filter by start_date and end_date
            if start_date and article_date < start_date:
                continue
            if end_date and article_date > end_date:
                continue

            news_item = {
                "Title": entry.title,
                "URL": entry.link,
                "Date": article_date.strftime("%Y-%m-%d %H:%M"),
                "Summary": entry.summary if "summary" in entry else "",
                "Media": "Bloomberg",
                "Image": (
                    entry.media_content[0]["url"] if "media_content" in entry else None
                ),
            }
            news_list.append(news_item)

        return news_list

    except Exception as e:
        print(f"Error fetching Bloomberg News: {e}")
        return []


# Combine News from Multiple Sources
def analyze_combined_news(query):
    """
    Combine Yahoo Finance and NewsAPI articles for the given stock query.
    """
    try:
        google_news = fetch_google_news(query)
    except Exception as e:
        print(f"Error fetching Yahoo Finance news for '{query}': {e}")
        google_news = pd.DataFrame(
            columns=["Title", "URL", "Date", "Summary", "Sentiment"]
        )

    # Combine the results
    combined_news = pd.concat([google_news], ignore_index=True)

    return combined_news.to_dict(orient="records")
