import schedule
import time
import pytz
from datetime import datetime
from app import create_app
from app.utils.newsapi_fetcher import fetch_newsapi_articles
from app.utils.yahoo_news_scraper import scrape_yahoo_finance_news
from app.models import Symbol  # Fetch stock symbols dynamically

# Load Flask app
app = create_app()

# Set timezone for scheduling (e.g., New York for US market)
MARKET_TIMEZONE = pytz.timezone("America/New_York")

# **Set scraping times based on market activity**
SCRAPE_TIMES = ["09:00", "18:00"]  # 9 AM & 6 PM Market time


def get_tracked_symbols():
    """Fetch all stock symbols tracked in the database."""
    with app.app_context():
        return Symbol.get_all_symbols()  # Dynamically get symbols


def scrape_all_news():
    """
    Fetches stock-related news articles using multiple sources.
    Stores articles in MongoDB and performs sentiment analysis.
    """
    with app.app_context():
        symbols = get_tracked_symbols()
        if not symbols:
            print("⚠️ No symbols found in the database!")
            return

        now = datetime.now(MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔄 Running scheduled news scraping at {now}")

        for symbol in symbols:
            try:
                print(f"🔍 Scraping news for: {symbol}")
                fetch_newsapi_articles(symbol)  # Fetch from NewsAPI
                scrape_yahoo_finance_news(symbol)  # Scrape from Yahoo Finance
            except Exception as e:
                print(f"⚠️ Error scraping news for {symbol}: {e}")

        print(f"✅ News scraping completed at {datetime.now(MARKET_TIMEZONE)}")


def set_schedule():
    """Schedule news scraping tasks dynamically based on timezone."""
    for scrape_time in SCRAPE_TIMES:
        schedule.every().day.at(scrape_time, MARKET_TIMEZONE).do(scrape_all_news)

    print(f"⏳ News scraping scheduled at {SCRAPE_TIMES} {MARKET_TIMEZONE}")


def run_scheduler():
    """Run the scheduler to execute tasks at the scheduled times."""
    with app.app_context():
        set_schedule()
        print("🚀 News scraping scheduler started!")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Wait 1 min before checking again


if __name__ == "__main__":
    run_scheduler()
