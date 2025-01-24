import schedule
import time
from yahoo_news_scraper import scrape_yahoo_finance_news


def scheduled_scraping():
    """
    Schedule the scraping task for specific stock symbols.
    """
    stock_symbols = ["AAPL", "MSFT", "GOOG"]  # Add more symbols as needed
    for symbol in stock_symbols:
        scrape_yahoo_finance_news(symbol)


# Schedule the scraping task to run every 90 minutes
schedule.every(90).minutes.do(scheduled_scraping)

print("Scheduler is running. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(1)
