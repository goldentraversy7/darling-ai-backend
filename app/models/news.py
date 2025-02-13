from app.extensions import mongo


class News:
    @staticmethod
    def save_news_to_mongo(news_data):
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
