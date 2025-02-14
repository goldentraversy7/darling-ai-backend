import torch
import pickle
import numpy as np
from app.ai.stock_transformer import StockTransformer
from app.services.stock_service import StockService


def predict_next_3_days(symbol):
    """
    Uses fine-tuned Transformer model to predict stock prices for next 3 days.
    """
    input_size = 9
    model = StockTransformer(input_size)
    model.load_state_dict(torch.load("models_storage/fine_tuned_transformer.pth"))
    model.eval()

    with open(f"models_storage/scaler_{symbol}.pkl", "rb") as f:
        scaler = pickle.load(f)

    recent_stock_data = StockService.merge_sentiment_with_stock(symbol).tail(30)
    features = [
        "close",
        "SMA_20",
        "EMA_20",
        "RSI",
        "MACD",
        "BB_Upper",
        "BB_Lower",
        "sentiment",
        "symbol_encoded",
    ]
    recent_stock_data_scaled = scaler.transform(recent_stock_data[features])

    input_tensor = torch.tensor([recent_stock_data_scaled], dtype=torch.float32)
    predictions = model(input_tensor).detach().numpy()
    predictions = scaler.inverse_transform(predictions)

    return predictions
