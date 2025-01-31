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
import tweepy
import nltk

nltk.download("punkt")
# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
BING_API_KEY = os.getenv("BING_SEARCH_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")

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
            # bloomberg_news = fetch_bloomberg_news(query, start_date, end_date)

            # Create response dictionary
            response = {
                "google_news": google_news,
                "bing_news": bing_news,
                "yahoo_news": yahoo_news,
                # "bloomberg_news": bloomberg_news,
            }

            return response  # Convert response to JSON
        except Exception as e:
            print(f"Unexpected error: {e}")
            return json.dumps({"error": f"Unexpected error: {str(e)}"})

    @staticmethod
    def fetch_socialmedia_posts(query, start_date=None, end_date=None):
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
            twitter_posts = fetch_twitter_posts(query, start_date, end_date)
            linkedin_posts = fetch_linkedin_posts(query)
            instagram_posts = fetch_instagram_posts(query)

            # Create response dictionary
            response = {
                "twitter": twitter_posts,
                "linkedin": linkedin_posts,
                "instagram": instagram_posts,
            }

            return response  # Convert response to JSON
        except Exception as e:
            print(f"Unexpected error: {e}")
            return json.dumps({"error": f"Unexpected error: {str(e)}"})

    @staticmethod
    def fetch_public_records(query, start_date=None, end_date=None, count=50):
        """Fetch public records: criminal records, court cases, and legal reports."""
        try:
            today = datetime.today().date()
            if end_date is None or end_date > today:
                end_date = today
            if start_date is None or start_date > today:
                start_date = today - timedelta(days=7)

            court_records = fetch_us_court_records(query, start_date, end_date, count)
            # criminal_records = fetch_criminal_records(query, count)
            # inmate_records = fetch_inmate_records(query, count)
            # business_records = fetch_business_records(query, count)

            response = {
                "court_records": court_records,
                # "criminal_records": criminal_records,
                # "inmate_records": inmate_records,
                # "business_records": business_records,
            }
            return response
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
        print(df.columns)

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


def fetch_yahoo_news(query, start_date=None, end_date=None):
    try:
        rss_url = f"https://news.search.yahoo.com/rss?p={query}"
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


def fetch_twitter_posts(query, start_date=None, end_date=None, max_results=20):
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

        # Twitter API requires ISO format for date
        if start_date:
            start_date = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end_date:
            end_date = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fetch tweets
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            start_time=start_date,
            end_time=end_date,
            tweet_fields=["created_at", "text", "author_id"],
        )

        tweets = []
        for tweet in response.data:
            tweets.append(
                {
                    "Text": tweet.text,
                    "URL": f"https://twitter.com/user/status/{tweet.id}",
                    "Date": datetime.datetime.strptime(
                        tweet.created_at, "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).strftime("%Y-%m-%d %H:%M"),
                    "Author_ID": tweet.author_id,
                }
            )

        return tweets

    except Exception as e:
        print(f"Error fetching Twitter posts: {e}")
        return []


def fetch_linkedin_posts(query, count=10):
    try:
        headers = {
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        url = f"https://api.linkedin.com/v2/shares?q=recent&q=SEARCH_TERM:{query}&count={count}"
        response = requests.get(url, headers=headers)
        data = response.json()

        posts = []
        for item in data.get("elements", []):
            posts.append(
                {
                    "Text": item["specificContent"]["com.linkedin.ugc.ShareContent"][
                        "shareCommentary"
                    ]["text"],
                    "URL": item.get("id", ""),
                    "Date": datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),  # LinkedIn API doesn't provide exact timestamps
                    "Media": "LinkedIn",
                }
            )

        return posts

    except Exception as e:
        print(f"Error fetching LinkedIn posts: {e}")
        return []


def fetch_instagram_posts(query, count=10):
    try:
        url = f"https://graph.facebook.com/v18.0/{INSTAGRAM_ACCOUNT_ID}/media?fields=id,caption,media_url,timestamp,permalink&limit={count}&access_token={INSTAGRAM_ACCESS_TOKEN}"
        response = requests.get(url)
        data = response.json()

        posts = []
        for item in data.get("data", []):
            posts.append(
                {
                    "Text": item.get("caption", ""),
                    "URL": item.get("permalink", ""),
                    "Date": datetime.datetime.strptime(
                        item["timestamp"], "%Y-%m-%dT%H:%M:%S%z"
                    ).strftime("%Y-%m-%d %H:%M"),
                    "Image": item.get("media_url", ""),
                    "Media": "Instagram",
                }
            )

        return posts

    except Exception as e:
        print(f"Error fetching Instagram posts: {e}")
        return []


def fetch_us_court_records(query, start_date=None, end_date=None, count=50):
    """Fetch US court cases from CourtListener API."""
    try:
        url = f"https://www.courtlistener.com/api/rest/v4/search/?q={query}&type=r"
        response = requests.get(url)
        data = response.json()

        records = []
        for item in data.get("results", [])[:count]:
            records.append(
                {
                    "Case Name": item.get("caseName", "No Case Name"),
                    "Court": item.get("court", "Unknown"),
                    "Filed Date": item.get("dateFiled", "Unknown"),
                    "URL": item.get("docket_absolute_url", ""),
                }
            )

        return records
    except Exception as e:
        print(f"Error fetching US court records: {e}")
        return []


def fetch_criminal_records(query, count=50):
    """Fetch criminal records from the National Sex Offender Registry."""
    try:
        url = f"https://www.nsopw.gov/api/Search?query={query}"
        response = requests.get(url)
        data = response.json()

        offenders = []
        for item in data.get("results", [])[:count]:
            offenders.append(
                {
                    "Name": item.get("name", "Unknown"),
                    "State": item.get("state", "Unknown"),
                    "Offense": item.get("offense", "Unknown"),
                    "URL": item.get("profileUrl", ""),
                }
            )

        return offenders
    except Exception as e:
        print(f"Error fetching criminal records: {e}")
        return []


def fetch_inmate_records(query, count=50):
    """Fetch inmate records from Bureau of Prisons API."""
    try:
        url = f"https://www.bop.gov/inmateloc/api/v1/inmates?lastName={query}"
        response = requests.get(url)
        data = response.json()

        inmates = []
        for item in data.get("inmates", [])[:count]:
            inmates.append(
                {
                    "Name": item.get("fullName", "Unknown"),
                    "Age": item.get("age", "Unknown"),
                    "Release Date": item.get("releaseDate", "Unknown"),
                    "Facility": item.get("facility", "Unknown"),
                    "URL": item.get("inmateURL", ""),
                }
            )

        return inmates
    except Exception as e:
        print(f"Error fetching inmate records: {e}")
        return []


def fetch_business_records(query, count=50):
    """Fetch business records from OpenCorporates."""
    try:
        url = f"https://api.opencorporates.com/v0.4/companies/search?q={query}"
        response = requests.get(url)
        data = response.json()

        businesses = []
        for item in data.get("results", {}).get("companies", [])[:count]:
            businesses.append(
                {
                    "Company Name": item.get("company", {}).get("name", "Unknown"),
                    "Jurisdiction": item.get("company", {}).get(
                        "jurisdiction_code", "Unknown"
                    ),
                    "Company Number": item.get("company", {}).get(
                        "company_number", "Unknown"
                    ),
                    "Status": item.get("company", {}).get("current_status", "Unknown"),
                    "URL": item.get("company", {}).get("opencorporates_url", ""),
                }
            )

        return businesses
    except Exception as e:
        print(f"Error fetching business records: {e}")
        return []
