import pandas as pd
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import re
from utils import analyze_sentiment

# Global Configurations
CHROMEDRIVER_PATH = os.path.abspath("../../chromedriver-win64/chromedriver.exe")
CSV_FILE_PATH = os.path.abspath("../../yahoo_news_data.csv")


def scrape_yahoo_finance_news(symbol):
    """
    Scrape Yahoo Finance news articles dynamically loaded by JavaScript.
    Save the scraped data to a CSV file if not already present.
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
    new_data = pd.DataFrame(
        articles,
        columns=["Symbol", "Title", "URL", "Date", "Summary", "Sentiment"],
    )
    new_data = new_data.iloc[::-1]

    # Check if CSV exists
    if os.path.exists(CSV_FILE_PATH):
        existing_data = pd.read_csv(CSV_FILE_PATH)

        # Filter the last 50 news for the same symbol
        last_50 = existing_data[existing_data["Symbol"] == symbol].tail(50)

        # Remove duplicates by checking against the last 50 news
        new_data = new_data[
            ~new_data["Title"].isin(last_50["Title"])
            & ~new_data["URL"].isin(last_50["URL"])
        ]

    if not new_data.empty:
        # Append only new data to the CSV
        new_data.to_csv(
            CSV_FILE_PATH,
            mode="a",
            header=not os.path.exists(CSV_FILE_PATH),
            index=False,
        )
        print(f"Saved {len(new_data)} new articles for {symbol} to {CSV_FILE_PATH}")
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


# Example usage for testing
if __name__ == "__main__":
    scrape_yahoo_finance_news("AAPL")
