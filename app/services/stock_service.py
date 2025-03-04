import os
import time
import json
import base64
import requests
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from dotenv import load_dotenv
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from app.models import News  # Import MongoDB save function
from app.ai.predict_lstm import predict_stock_price

# Load environment variables from .env file
load_dotenv()

# Load NewsAPI Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Global Configurations
pd.set_option("display.max_colwidth", 1000)
CSV_FILE_PATH = os.path.abspath("./yahoo_news_data.csv")


class StockService:
    """
    Service class to fetch stock data from various sources.
    """

    @staticmethod
    def fetch_stock_data(symbol, company=""):
        """
        Fetch stock data from multiple sources for a given stock symbol.
        """
        try:

            # Analyze stock data and generate the plot
            news_articles = StockService.fetch_news_articles(symbol)
            stock_data = StockService.analyze_stock_data(symbol)
            predicted_prices = predict_stock_price(symbol)
            predictions = []

            if predicted_prices is not None and len(predicted_prices) == 3:
                latest_date = datetime.utcnow().date()

                # Append predictions as new entries
                for i, price in enumerate(predicted_prices):
                    future_date = latest_date + timedelta(days=i + 1)
                    predictions.append(
                        {
                            "dDate": future_date.strftime(
                                "%Y-%m-%d"
                            ),  # Ensure date is a string
                            "close": float(price),  # Convert np.float64 to native float
                            "Predicted": True,  # Flag as predicted data
                        }
                    )

            # Create response dictionary
            response = {
                "stock_data": stock_data,
                "news_articles": news_articles,
                "predictions": predictions,
            }

            return response  # Convert response to JSON
        except Exception as e:
            print(f"Unexpected error: {e}")
            return json.dumps({"error": f"Unexpected error: {str(e)}"})

    @staticmethod
    def fetch_news_articles(symbol, period="1y"):
        try:
            articles = News.fetch_news_from_db(symbol, period)
            return articles
        except Exception as e:
            print(f"Error fetching Yahoo Finance news from CSV for '{symbol}': {e}")
            return []

    @staticmethod
    def analyze_stock_data(symbol, period="1y"):
        try:
            # Fetch stock data
            stock_data = fetch_stock_data(symbol, period=period)

            # Generate the plot
            stock_data["dDate"] = stock_data.index
            # Return the analyzed data and plot
            # Handle NaN values and convert DataFrame to JSON-serializable format
            if stock_data is not None:
                # Replace NaN with None (which converts to null in JSON)
                stock_data = stock_data.replace(
                    [np.nan, np.inf, -np.inf], None
                ).to_dict(orient="records")
            else:
                stock_data = []

            return stock_data

        except Exception as e:
            # Log the error for debugging
            print(f"Error in analyzing stock data for symbol '{symbol}': {e}")

            # Return empty values or default placeholders
            return None

    @staticmethod
    def merge_sentiment_with_stock(symbol, period="1y"):
        stock_data = StockService.analyze_stock_data(symbol, period)
        news_data = StockService.fetch_news_articles(symbol, period)

        # Convert stock data to DataFrame (Ensure Date is in datetime format)
        stock_df = pd.DataFrame(stock_data)
        stock_df["dDate"] = pd.to_datetime(stock_df["dDate"])  # Convert Date column

        # Convert news data to DataFrame (Extract only relevant columns)
        news_df = pd.DataFrame(news_data, columns=["dDate", "Sentiment"])
        news_df["dDate"] = pd.to_datetime(news_df["dDate"])  # Convert to datetime

        # Aggregate daily sentiment score
        daily_sentiment = news_df.groupby("dDate")["Sentiment"].mean().reset_index()

        # Merge stock data with daily sentiment
        merged_data = stock_df.merge(
            daily_sentiment, left_on="dDate", right_on="dDate", how="left"
        )

        # Fill missing sentiment values with 0 (no news = neutral sentiment)
        merged_data.loc[:, "Sentiment"] = merged_data["Sentiment"].fillna(0)

        return merged_data


def calculate_technical_indicators(df):
    """
    Calculates additional technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands.
    """
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD Calculation
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Upper"] = df["SMA_20"] + (df["close"].rolling(window=20).std() * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["close"].rolling(window=20).std() * 2)

    return df


def fetch_stock_data(symbol, interval="1d", period="1y"):
    """
    Fetch historical stock data using Yahoo Finance (yfinance).

    :param symbol: Stock ticker symbol.
    :param interval: Interval of stock data (e.g., "1d", "1wk", "1mo").
    :param period: How much history to fetch (e.g., "6mo", "1y", "5y").
    :return: Pandas DataFrame with stock data and indicators.
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return None

        # Keep relevant columns and rename
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Add stock symbol as a feature
        df["Symbol"] = symbol

        # Calculate additional technical indicators
        df = calculate_technical_indicators(df)

        # Convert datetime index to string for JSON response
        df.index = df.index.strftime("%Y-%m-%d")

        return df

    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None
