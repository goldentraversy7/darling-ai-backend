from app import create_app
import torch
import numpy as np
import pickle
from app.ai.stock_lstm import StockLSTM
from app.services.stock_service import StockService
from app.ai.rl_reward import calculate_reward
import os

# Initialize Flask app
app = create_app()

# File Paths
MODEL_PATH = "models_storage/best_lstm_model.pth"
SCALER_PATH = "models_storage/scaler.pkl"
ENCODER_PATH = "models_storage/label_encoder.pkl"


def update_model_with_actual_data(symbol):
    """
    Fetches actual stock prices, computes reward, and updates the LSTM model.
    """
    # **Load trained model (if available)**
    input_size = 9
    model = StockLSTM(input_size)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded existing model from {MODEL_PATH}")
    else:
        print("No pre-trained model found. Train the model first.")
        return

    model.train()  # Set to training mode

    # **Fetch latest stock data**
    merged_data = StockService.merge_sentiment_with_stock(symbol)

    # **Load Scaler & Encoder**
    if not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
        print("Missing scaler or encoder! Run training first.")
        return

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    # **Ensure Symbol Encoding**
    if "Symbol" not in merged_data.columns:
        merged_data["Symbol"] = symbol

    if symbol not in label_encoder.classes_:
        print(f"New stock symbol {symbol} detected! Encoding dynamically.")
        label_encoder.classes_ = np.append(label_encoder.classes_, symbol)

    merged_data["symbol_encoded"] = label_encoder.transform(merged_data["Symbol"])

    # **Select features for scaling**
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

    # **Ensure features exist before scaling**
    missing_features = [f for f in features if f not in merged_data.columns]
    if missing_features:
        print(f"Missing features in merged_data: {missing_features}")
        return

    scaled_data = scaler.transform(merged_data[features])

    # **Prepare input data (Last 60 days)**
    input_data = np.array([scaled_data[-63:-3]])  # Exclude last 3 days
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # **Fetch actual closing prices (Last 3 days)**
    actual_prices = merged_data["close"].values[-3:]

    # **Predict next 3 days**
    with torch.no_grad():
        predicted_prices = model(input_tensor).numpy().flatten()

    # **Compute RL Reward**
    reward = calculate_reward(predicted_prices, actual_prices)
    print(f"RL Reward for {symbol}: {reward}")

    # **Fine-Tune Model with New Data**
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

    # **Save Updated Model**
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model updated with new data for {symbol}")


if __name__ == "__main__":
    with app.app_context():
        update_model_with_actual_data("AAPL")
