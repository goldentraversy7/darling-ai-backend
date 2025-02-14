import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.ai.predict_lstm import predict_next_3_days as predict_lstm
from app.ai.predict_transformer import predict_next_3_days as predict_transformer
from app.services.stock_service import StockService


def evaluate_models(symbol):
    """
    Evaluates LSTM and Transformer models by comparing their predictions
    with actual stock prices.
    """
    # Get actual stock prices for the next 3 days
    stock_data = StockService.merge_sentiment_with_stock(symbol).tail(
        33
    )  # Last 33 days
    actual_prices = stock_data["close"].values[-3:]  # Get last 3 days as ground truth

    # Get predictions
    lstm_predictions = predict_lstm(symbol)
    transformer_predictions = predict_transformer(symbol)

    # Ensure predictions match actual shape
    lstm_predictions = np.array(lstm_predictions).flatten()
    transformer_predictions = np.array(transformer_predictions).flatten()

    # Compute errors
    lstm_mse = mean_squared_error(actual_prices, lstm_predictions)
    lstm_mae = mean_absolute_error(actual_prices, lstm_predictions)

    transformer_mse = mean_squared_error(actual_prices, transformer_predictions)
    transformer_mae = mean_absolute_error(actual_prices, transformer_predictions)

    # Print results
    print(f"📊 **Evaluation Results for {symbol}:**")
    print(f"LSTM - MSE: {lstm_mse:.4f}, MAE: {lstm_mae:.4f}")
    print(f"Transformer - MSE: {transformer_mse:.4f}, MAE: {transformer_mae:.4f}")

    # Return as dictionary
    return {
        "LSTM": {"MSE": lstm_mse, "MAE": lstm_mae},
        "Transformer": {"MSE": transformer_mse, "MAE": transformer_mae},
    }


# Run evaluation
if __name__ == "__main__":
    symbol = "AAPL"  # Example symbol
    results = evaluate_models(symbol)
