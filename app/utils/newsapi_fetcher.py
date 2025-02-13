import os
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
from app import create_app
from app.models import News  # Import MongoDB save function
from app.utils.utils import analyze_sentiment  # Import sentiment analysis

# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize Flask app
app = create_app()

# Global Configurations
pd.set_option("display.max_colwidth", 1000)


def fetch_newsapi_articles(symbol):
    """
    Fetch articles from NewsAPI for a given symbol and date range.
    Save the scraped data to MongoDB.
    """
    sources = fetch_newsapi_sources()
    today = datetime.today().strftime("%d-%b-%Y")
    return fetch_articles_with_sentiments(symbol, today, sources)


def fetch_newsapi_sources():
    """
    Fetch top news sources from NewsAPI related to stocks & finance.
    """
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    sources_data = newsapi.get_sources()

    # List of financial & stock market-related sources
    stock_related_sources = [
        "bloomberg",
        "business-insider",
        "cnbc",
        "financial-post",
        "financial-times",
        "fortune",
        "marketwatch",
        "the-wall-street-journal",
        "the-economist",
    ]

    # Filter sources based on category (business + general) OR specific financial sources
    sources = [
        source["id"]
        for source in sources_data["sources"]
        if source["category"] in ["business", "general"]
        or source["id"] in stock_related_sources
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
    from_date = start_date - timedelta(days=30)

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
        return pd.DataFrame(
            columns=["Symbol", "Title", "URL", "Date", "Summary", "Sentiment"]
        )

    # Process articles
    seen_titles = set()
    articles_data = []

    for article in articles.get("articles", []):
        if article["title"] in seen_titles:
            continue
        seen_titles.add(article["title"])
        content = f"{article['title']}. {article['description']}".lower()
        if symbol.lower() not in content:
            continue
        sentiment = analyze_sentiment(content)
        date = article.get("publishedAt", "No date available")
        date = (
            datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
            if date != "No date available"
            else date
        )

        articles_data.append(
            (
                symbol,
                article["title"],
                article["url"],
                date,
                article["description"],
                sentiment,
            )
        )

    # Convert to DataFrame
    news_df = pd.DataFrame(
        articles_data,
        columns=["Symbol", "Title", "URL", "dDate", "Summary", "Sentiment"],
    )

    if not news_df.empty:
        with app.app_context():  # ✅ Ensure the MongoDB connection is active
            News.save_news_to_db(news_df.to_dict(orient="records"))
    else:
        print(f"No new articles found for {symbol}")


# ✅ Run the fetcher inside Flask app context
if __name__ == "__main__":
    with app.app_context():  # Ensure MongoDB is initialized before running
        fetch_newsapi_articles("AAPL")
