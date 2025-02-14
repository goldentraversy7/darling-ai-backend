from app import create_app
import torch
import pickle
import numpy as np
from app.ai.stock_lstm import StockLSTM
from app.services.stock_service import StockService
import threading
from app.ai.train_lstm import fine_tune_lstm

# Initialize Flask app
app = create_app()


# Paths to stored model and scalers
MODEL_PATH = "models_storage/best_lstm_model.pth"
SCALER_PATH = "models_storage/scaler.pkl"
ENCODER_PATH = "models_storage/label_encoder.pkl"


def load_trained_lstm():
    """
    Load the trained LSTM model for stock prediction.
    """
    input_size = 9  # Number of input features
    model = StockLSTM(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def prepare_data_for_prediction(symbol, lookback=60):
    """
    Fetch latest stock data, process it, and prepare for LSTM model input.
    """
    merged_data = StockService.merge_sentiment_with_stock(symbol)

    # Load scaler and label encoder
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    # Encode stock symbol (handle new symbols dynamically)
    if symbol not in label_encoder.classes_:
        print(f"New stock symbol {symbol} detected! Encoding as new class.")
        label_encoder.classes_ = np.append(label_encoder.classes_, symbol)

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
    If it's new, start fine-tuning in the background.
    """
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    if symbol not in label_encoder.classes_:
        print(f"🚀 New stock detected: {symbol}. Fine-tuning model...")
        threading.Thread(target=fine_tune_lstm, args=([symbol],)).start()


#  Run the scraper inside Flask app context
if __name__ == "__main__":
    with app.app_context():
        predict_stock_price("AAPL")
