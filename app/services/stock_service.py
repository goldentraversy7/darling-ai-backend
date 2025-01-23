import os
import requests
from requests.auth import HTTPBasicAuth
import requests
from datetime import datetime, timedelta
import json
from bs4 import BeautifulSoup
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import yfinance as yf
from yahooquery import Ticker

class StockService:
    @staticmethod
    def fetch_stock_data(symbol, company = ''):
        """
        Fetch stock data from multiple websites.
        """
        try:
            news_articles = scrape_yahoo_finance_news(symbol)
            
            return news_articles
        except requests.exceptions.RequestException as e:
            print(f"Error fetching order history: {e}")
            return {"error": str(e)}
        

def scrape_yahoo_finance_news(stock_symbol):
    """
    Scrape news articles dynamically loaded by JavaScript on Yahoo Finance.
    """
    # Setup Selenium WebDriver
    cService = webdriver.ChromeService(
        executable_path="./chromedriver-win64/chromedriver.exe"
    )
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-software-rasterizer")  # Disable software rasterizer
    options.add_argument("--use-gl=swiftshader")
    options.add_argument(
        "--start-maximized"
    )  # Start browser maximized for easier UI interaction
    driver = webdriver.Chrome(service=cService, options=options)

    url = f"https://finance.yahoo.com/quote/{stock_symbol}/news"
    driver.get(url)

    # Allow time for the JavaScript to load
    time.sleep(2)  # Adjust this delay if needed

    # Fetch the rendered page source
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()  # Close the browser session

    # Parse the news section
    news_items = soup.select('section[data-testid="storyitem"]')  # Adjusted based on Yahoo layout

    news = []
    for idx, item in enumerate(news_items, start=1):
        # Extract the article link
        link_elem = item.select_one('a.subtle-link[href]')
        link = link_elem['href'] if link_elem else None
        if link and link.startswith('/'):
            link = f"https://finance.yahoo.com{link}"  # Handle relative links

        # Extract the article title
        title_elem = item.select_one('h3')
        title = title_elem.get_text(strip=True) if title_elem else None

        # Extract the publishing date
        date_elem = item.select_one('div.publishing')
        date = date_elem.get_text(strip=True) if date_elem else "No date available"

        # Extract the summary
        summary_elem = item.select_one('p')
        summary = summary_elem.get_text(strip=True) if summary_elem else "No summary available"

        if title and link:
            news.append({
                "title": title,
                "link": link,
                "date": date,
                "summary": summary
            })

    print(f"Total articles scraped for {stock_symbol}: {len(news)}")
    return news
