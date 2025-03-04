import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import current_app

# Paths
MODEL_DIR = os.path.abspath("models_storage")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def get_app():
    """Safely get Flask app context to prevent circular imports."""
    from app import create_app  # Moved import inside function

    return create_app()


def prepare_stock_data(stock_symbols):
    """
    Fetch stock data with sentiment for multiple stocks and prepare it for LSTM training.
    """
    from app.services.stock_service import StockService  # Lazy import to prevent issues

    with current_app.app_context():  # Ensure Flask app context
        all_data = []

        # **Load existing label encoder (if available)**
        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)
            print(f" Loaded existing label encoder: {label_encoder.classes_}")
        else:
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array([])  # Start empty
            print(" No label encoder found. Creating a new one.")

        # Ensure models_storage directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Collect stock data
        for symbol in stock_symbols:
            merged_data = StockService.merge_sentiment_with_stock(symbol, period="3y")
            merged_data["Symbol"] = symbol  # Ensure Symbol column exists
            all_data.append(merged_data)

        # Combine all stock data
        full_data = pd.concat(all_data, ignore_index=True)

        # **Merge new symbols while preserving order**
        existing_symbols = list(label_encoder.classes_)  # Convert to list
        unique_symbols = list(full_data["Symbol"].unique())

        # Append only new symbols without modifying order
        for symbol in unique_symbols:
            if symbol not in existing_symbols:
                existing_symbols.append(symbol)

        # Convert back to numpy array
        label_encoder.classes_ = np.array(existing_symbols)

        print(f"Updated label encoder: {label_encoder.classes_}")

        # **Use .transform() to prevent overwriting existing mappings**
        full_data["symbol_encoded"] = label_encoder.transform(full_data["Symbol"])

        # **Ensure label encoder is always saved persistently**
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(label_encoder, f)
            print(f"Saved label encoder: {label_encoder.classes_}")

        # Feature selection
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

        # Handle missing values
        for col in ["SMA_20", "EMA_20", "RSI", "MACD", "BB_Upper", "BB_Lower"]:
            full_data.loc[:, col] = full_data[col].ffill()  # Use recommended `ffill()`
            full_data.loc[:, col] = full_data[col].fillna(
                full_data[col].mean()
            )  # Fill remaining NaN

        # **Fixing Sentiment NaN Issue**
        full_data.loc[:, "Sentiment"] = full_data["Sentiment"].fillna(
            0.0
        )  # No news = Neutral sentiment

        full_data.dropna(subset=["close"], inplace=True)

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(full_data[features])

        # **Save scaler**
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        # Create sequences for LSTM
        X, y = [], []
        context_length, prediction_length = 60, 3

        for i in range(len(scaled_data) - context_length - prediction_length):
            X.append(scaled_data[i : i + context_length])
            y.append(
                scaled_data[
                    i + context_length : i + context_length + prediction_length, 0
                ]
            )

        return np.array(X), np.array(y)


if __name__ == "__main__":
    app = get_app()  # Initialize Flask app
    with app.app_context():
        prepare_stock_data(
            ["AAPL", "GOOGL", "MSFT", "TSLA"]
        )  # Train on multiple stocks
