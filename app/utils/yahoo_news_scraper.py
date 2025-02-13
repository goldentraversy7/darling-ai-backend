from app import create_app
import pandas as pd
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import re

from app.models import News  # Import MongoDB save function
from app.utils.utils import analyze_sentiment

# Initialize Flask app
app = create_app()

# Global Configurations
CHROMEDRIVER_PATH = os.path.abspath("./chromedriver-win64/chromedriver.exe")


def scrape_yahoo_finance_news(symbol):
    """
    Scrape Yahoo Finance news articles dynamically loaded by JavaScript.
    Save the scraped data to MongoDB if not already present.
    """
    # Configure Selenium WebDriver
    chrome_service = webdriver.ChromeService(executable_path=CHROMEDRIVER_PATH)
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--use-gl=swiftshader")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(service=chrome_service, options=options)

    # Open Yahoo Finance news page
    url = f"https://finance.yahoo.com/quote/{symbol}/latest-news"
    driver.get(url)

    # Allow time for JavaScript to load
    driver.implicitly_wait(5)

    # Parse the page
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Extract news articles
    news_items = soup.select('section[data-testid="storyitem"]')
    articles = []

    for item in news_items:
        # Extract article details
        title = (
            item.select_one("h3").get_text(strip=True)
            if item.select_one("h3")
            else None
        )
        link_elem = item.select_one("a.subtle-link[href]")
        link = (
            f"https://finance.yahoo.com{link_elem['href']}"
            if link_elem and link_elem["href"].startswith("/")
            else link_elem["href"] if link_elem else None
        )
        date_text = (
            item.select_one("div.publishing").get_text(strip=True)
            if item.select_one("div.publishing")
            else "No date available"
        )
        summary = (
            item.select_one("p").get_text(strip=True)
            if item.select_one("p")
            else "No summary available"
        )

        if title and link:
            utc_date = parse_relative_date(date_text)
            sentiment = analyze_sentiment(f"{title} {summary}")
            articles.append((symbol, title, link, utc_date, summary, sentiment))

    # Convert to DataFrame
    news_df = pd.DataFrame(
        articles,
        columns=["Symbol", "Title", "URL", "dDate", "Summary", "Sentiment"],
    )

    if not news_df.empty:
        with app.app_context():  # ✅ Ensure the MongoDB connection is active
            News.save_news_to_db(news_df.to_dict(orient="records"))
    else:
        print(f"No new articles found for {symbol}")


def parse_relative_date(date_text):
    """
    Parse relative date text (e.g., '9 hours ago', 'Yahoo•9 hours ago') into a UTC datetime string.
    """
    now = datetime.now(timezone.utc)  # Use timezone-aware UTC datetime

    # Extract the relative time part using regex
    match = re.search(r"(\d+)\s+(hours|minutes|days)\s+ago", date_text, re.IGNORECASE)
    if match:
        value, unit = int(match.group(1)), match.group(2).lower()

        # Adjust the current time based on the relative unit
        if unit == "hours":
            return (now - timedelta(hours=value)).strftime("%Y-%m-%d")
        elif unit == "minutes":
            return (now - timedelta(minutes=value)).strftime("%Y-%m-%d")
        elif unit == "days":
            return (now - timedelta(days=value)).strftime("%Y-%m-%d")

    # If no match is found, return the current UTC time as a fallback
    return now.strftime("%Y-%m-%d")


# ✅ Run the scraper inside Flask app context
if __name__ == "__main__":
    with app.app_context():  # Ensure MongoDB is initialized before running
        scrape_yahoo_finance_news("AAPL")
