import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import current_app


def get_app():
    """Get the Flask app context safely to avoid circular import."""
    from app import (
        create_app,
    )  # Moved import inside function to prevent circular import

    return create_app()


def prepare_stock_data(stock_symbols):
    """
    Fetch stock data with sentiment for multiple stocks and prepare it for LSTM training.
    """
    from app.services.stock_service import (
        StockService,
    )  # Lazy import to prevent circular import

    with current_app.app_context():  # Ensures Flask app context is active
        all_data = []
        label_encoder = LabelEncoder()

        # Ensure storage directories exist
        os.makedirs("models_storage", exist_ok=True)

        # Collect data for all stocks
        for symbol in stock_symbols:
            merged_data = StockService.merge_sentiment_with_stock(
                symbol=symbol, period="3y"
            )
            merged_data["Symbol"] = symbol  # Ensure symbol column exists
            all_data.append(merged_data)

        # Combine all stock data into one DataFrame
        full_data = pd.concat(all_data, ignore_index=True)

        # Encode stock symbols as categorical feature
        full_data["symbol_encoded"] = label_encoder.fit_transform(full_data["Symbol"])

        # Select features for training
        features = [
            "close",
            "SMA_20",
            "EMA_20",
            "RSI",
            "MACD",
            "BB_Upper",
            "BB_Lower",
            "Sentiment",
            "symbol_encoded",
        ]

        # **Fixing Chained Assignment Warning & NaN Handling**
        for col in ["SMA_20", "EMA_20", "RSI", "MACD", "BB_Upper", "BB_Lower"]:
            full_data.loc[:, col] = full_data[col].ffill()  # Use recommended `ffill()`
            full_data.loc[:, col] = full_data[col].fillna(
                full_data[col].mean()
            )  # Fill remaining NaN with mean

        # **Fixing Sentiment NaN Issue**
        full_data.loc[:, "Sentiment"] = full_data["Sentiment"].fillna(
            0.0
        )  # No news = Neutral sentiment

        # **Drop rows where "close" price is NaN**
        full_data.dropna(subset=["close"], inplace=True)

        # **Scale data**
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(full_data[features])

        # Save scaler & label encoder for inference
        with open("models_storage/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        with open("models_storage/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        # **Create sequences for LSTM**
        X, y = [], []
        context_length = 60  # Use last 60 days for training
        prediction_length = 3  # Predict next 3 days

        for i in range(len(scaled_data) - context_length - prediction_length):
            X.append(scaled_data[i : i + context_length])  # Last 60 days
            y.append(
                scaled_data[
                    i + context_length : i + context_length + prediction_length, 0
                ]
            )  # Predict next 3 days close price

        return np.array(X), np.array(y)


# Run inside Flask app context
if __name__ == "__main__":
    app = get_app()  # Initialize Flask app safely
    with app.app_context():
        prepare_stock_data(
            ["AAPL", "GOOGL", "MSFT", "TSLA"]
        )  # Train on multiple stocks
