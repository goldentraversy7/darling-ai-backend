import os
import torch
import numpy as np
import pickle
from flask import current_app
from app.ai.stock_lstm import StockLSTM
from app.services.stock_service import StockService
from app.ai.rl_reward import calculate_reward
from sklearn.preprocessing import LabelEncoder

# Paths
MODEL_DIR = os.path.abspath("models_storage")
MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def get_app():
    """Safely get Flask app context to prevent circular imports."""
    from app import create_app

    return create_app()


def update_model_with_actual_data(symbol):
    """
    Fetches actual stock prices, computes reward, and updates the LSTM model.
    Ensures `label_encoder.pkl` is **preserved** during RL fine-tuning.
    """

    app = get_app()
    with app.app_context():  # Ensures Flask app context is active
        merged_data = StockService.merge_sentiment_with_stock(symbol)

    # Load Existing Model (Avoids Overwriting)
    input_size = 9
    model = StockLSTM(input_size)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded Existing Model from {MODEL_PATH}")
    else:
        print(" No pre-trained model found! Train the model first.")
        return

    model.train()  # Set model to training mode

    # Load Existing Scaler & Label Encoder (Avoid Overwriting)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    else:
        print(" No scaler found! Train the model first.")
        return

    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"Loaded Existing Label Encoder: {label_encoder.classes_}")
    else:
        print(" No previous label encoder found! Creating a new one.")
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array([])

    # Ensure Symbol Encoding is Up-to-Date
    existing_symbols = list(label_encoder.classes_)
    if symbol not in existing_symbols:
        print(f" New stock symbol detected: {symbol}. Adding to label encoder...")
        label_encoder.classes_ = np.array(existing_symbols + [symbol])  # Append only

        # Save Updated Label Encoder
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(label_encoder, f)
        print(f"Label Encoder Updated & Saved: {label_encoder.classes_}")

    merged_data["symbol_encoded"] = label_encoder.transform(merged_data["Symbol"])

    # Select Features for Scaling
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

    # Ensure features exist before scaling
    missing_features = [f for f in features if f not in merged_data.columns]
    if missing_features:
        print(f" Missing features in merged_data: {missing_features}")
        return

    scaled_data = scaler.transform(merged_data[features])

    # Prepare input data (Last 60 days, Excluding last 3 days)
    input_data = np.array([scaled_data[-63:-3]])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Fetch actual closing prices (Last 3 days)
    actual_prices = merged_data["close"].values[-3:]

    # Predict next 3 days
    with torch.no_grad():
        predicted_prices = model(input_tensor).numpy().flatten()

    # Compute RL Reward
    reward = calculate_reward(predicted_prices, actual_prices)
    print(f"RL Reward for {symbol}: {reward}")

    # Fine-Tune Model with New Data
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    optimizer.zero_grad()
    predictions_tensor = torch.tensor(
        predicted_prices, dtype=torch.float32, requires_grad=True
    )
    actual_tensor = torch.tensor(actual_prices, dtype=torch.float32)

    loss = criterion(predictions_tensor, actual_tensor)
    loss.backward()
    optimizer.step()

    # Save Updated Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model updated with new data for {symbol}")


if __name__ == "__main__":
    app = get_app()  # Initialize Flask app safely
    with app.app_context():
        update_model_with_actual_data("AAPL")
