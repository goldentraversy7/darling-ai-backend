import os
import torch
import pickle
import numpy as np
import threading
from flask import current_app
from app.ai.stock_lstm import StockLSTM
from app.ai.train_lstm import fine_tune_lstm
from sklearn.preprocessing import LabelEncoder

# Paths to stored model and scalers
MODEL_DIR = os.path.abspath("models_storage")  # Ensure absolute path
MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def get_app():
    """Get the Flask app context safely to avoid circular import."""
    from app import (
        create_app,
    )  # Moved import inside function to prevent circular import

    return create_app()


def load_trained_lstm():
    """Load the trained LSTM model for stock prediction."""
    input_size = 9  # Number of input features
    model = StockLSTM(input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


def prepare_data_for_prediction(symbol, lookback=60):
    """
    Fetch latest stock data, process it, and prepare for LSTM model input.
    """
    from app.services import StockService

    with current_app.app_context():  # Ensures Flask app context is active
        merged_data = StockService.merge_sentiment_with_stock(symbol)

    # Load scaler and label encoder
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    # Log current classes
    print(f"Existing Encoded Symbols: {label_encoder.classes_}")

    # Handle new stock symbols
    if symbol not in label_encoder.classes_:
        print(f"New stock symbol {symbol} detected! Adding to label encoder...")

        # 🔥 **Update Label Encoder**
        label_encoder.classes_ = np.append(label_encoder.classes_, symbol)

        # **Save Updated Label Encoder**
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(label_encoder, f)

        print(f"Label Encoder Updated & Saved: {label_encoder.classes_}")

    # Apply encoding to DataFrame
    merged_data["symbol_encoded"] = label_encoder.transform(merged_data["Symbol"])

    # Select features for scaling
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

    # Scale data
    scaled_data = scaler.transform(merged_data[features])

    # Take the last `lookback` days for prediction
    last_days = scaled_data[-lookback:]

    return np.array([last_days])  # Return as 3D array (batch, time steps, features)


def predict_stock_price(symbol):
    """
    Predict stock price for the next 3 days using LSTM.
    """
    model = load_trained_lstm()
    model.eval()

    input_data = prepare_data_for_prediction(symbol)

    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        predicted_prices = model(input_tensor)

    # Convert predictions back to real stock prices
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    predicted_prices = predicted_prices.numpy()
    print("Predicted Tensor Shape:", predicted_prices.shape)  # Debugging

    # Ensure predicted_prices shape (1, 3) aligns with scaler feature count (9)
    num_features = scaler.n_features_in_  # Should be 9

    # **Fix Shape Mismatch**
    full_pred_data = np.zeros((1, num_features))
    full_pred_data[0, 0:3] = predicted_prices  # Fill only "close" price predictions

    print(
        "Full Pred Data Shape Before Inverse Transform:", full_pred_data.shape
    )  # Debugging

    # Apply inverse transformation
    original_scale_prices = scaler.inverse_transform(full_pred_data)[
        0, :3
    ]  # Extract "close"

    print("Predicted Stock Prices for Next 3 Days:", original_scale_prices)

    return original_scale_prices


def check_and_train_new_symbol(symbol):
    """
    Check if a stock symbol exists in the trained model.
    If it's new, add it to label_encoder, save it, and fine-tune the model.
    """
    # Load existing label encoder if it exists
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
    else:
        print("🚨 No label encoder found! Creating a new one.")
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array([])  # Initialize with empty array

    existing_classes = set(label_encoder.classes_)  # Convert to set for merging

    # Check if symbol is new
    if symbol not in existing_classes:
        print(f"🚀 New stock detected: {symbol}. Adding to label encoder...")

        # 🔥 **Update Label Encoder (Merge New + Existing Classes)**
        updated_classes = sorted(existing_classes | {symbol})  # Merge & Sort
        label_encoder.classes_ = np.array(updated_classes)  # Convert back to array

        # **Save Updated Label Encoder**
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(label_encoder, f)

        print(f"Label Encoder Updated: {label_encoder.classes_}")

        # **Train Model in Background**
        threading.Thread(target=fine_tune_lstm, args=([symbol],), daemon=True).start()
        return True  # Model needs training

    return False  # No training needed


# Run inside Flask app context
if __name__ == "__main__":
    app = get_app()  # Initialize Flask app safely
    with app.app_context():
        predict_stock_price("AAPL")
