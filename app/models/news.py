from app.extensions import mongo
from datetime import datetime, timedelta


class News:
    @staticmethod
    def save_news_to_db(news_data):
        """
        Save news data to MongoDB.
        Filters out duplicate news by Title and URL.
        """
        for news in news_data:
            # Ensure data is extracted correctly using dictionary keys
            symbol = news.get("Symbol")
            title = news.get("Title")
            url = news.get("URL")
            dDate = news.get("dDate")  # Ensure consistent field name
            summary = news.get("Summary")
            sentiment = news.get("Sentiment")

            # Validate that required fields exist
            if not symbol or not title or not url:
                print("Skipping entry due to missing data:", news)
                continue  # Skip this entry if essential fields are missing

            # Check if the news article already exists
            existing_news = mongo.db.news.find_one(
                {"Symbol": symbol, "Title": title, "URL": url}
            )

            if not existing_news:
                news_document = {
                    "Symbol": symbol,
                    "Title": title,
                    "URL": url,
                    "dDate": dDate,  # Corrected field name for consistency
                    "Summary": summary,
                    "Sentiment": sentiment,
                }
                mongo.db.news.insert_one(news_document)

        print(f"✅ Completed saving {len(news_data)} articles to MongoDB.")

    @staticmethod
    def fetch_news_from_db(symbol, period="1mo"):
        """
        Retrieve news articles from MongoDB based on symbol and period.

        :param symbol: Stock symbol (e.g., "AAPL")
        :param period: Time period (e.g., "1mo", "7d", "3mo")
        :return: List of news articles matching the criteria
        """
        # Convert period to date range
        now = datetime.utcnow()
        period_mapping = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
        }

        # Default to 1 month if period not found
        date_threshold = now - period_mapping.get(period, timedelta(days=30))

        # Query MongoDB for news related to the given symbol & date range
        news_cursor = mongo.db.news.find(
            {
                "Symbol": symbol,
                "dDate": {"$gte": date_threshold.strftime("%Y-%m-%d")},
            }
        ).sort(
            "dDate", -1
        )  # Sort by most recent first

        # Convert cursor to list of dictionaries
        # Convert cursor to list of dictionaries and fix ObjectId issue
        news_articles = []
        for article in news_cursor:
            article["_id"] = str(article["_id"])  # Convert ObjectId to string
            news_articles.append(article)

        print(
            f"✅ Fetched {len(news_articles)} articles for {symbol} in the last {period}."
        )
        return news_articles
