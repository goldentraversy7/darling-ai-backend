from app import create_app
from app.services import StockService  # Import MongoDB save function
from transformers import (
    TimeSeriesTransformerForPrediction,
    TimeSeriesTransformerConfig,
    Trainer,
    TrainingArguments,
)
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Initialize Flask app
app = create_app()


def load_pretrained_model():
    """
    Load a pre-trained Time-Series Transformer for stock forecasting.
    """
    model_name = (
        "huggingface/time-series-transformer-tourism-monthly"  # Use a valid model
    )
    model = TimeSeriesTransformerForPrediction.from_pretrained(model_name)

    # Configure model for predicting 3 days ahead
    config = TimeSeriesTransformerConfig(
        prediction_length=3,  # Predict next 3 days
        context_length=30,  # Use last 30 days of stock data
        num_time_features=10,  # Stock indicators + sentiment
    )
    model.config = config

    return model


def prepare_stock_data_for_transformer(symbol):
    """
    Merges stock data with sentiment and prepares it for model training.
    """
    # Fetch merged stock + sentiment data
    merged_data = StockService.merge_sentiment_with_stock(symbol)

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
    ]

    # Encode the stock symbol as a numerical value
    label_encoder = LabelEncoder()
    merged_data["symbol_encoded"] = label_encoder.fit_transform(merged_data["Symbol"])

    # Add symbol as a feature
    features.append("symbol_encoded")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_data[features])

    # Create time-series sequences (e.g., last 30 days → next 3 days)
    X, y = [], []
    context_length = 30
    prediction_length = 3

    for i in range(len(scaled_data) - context_length - prediction_length):
        X.append(scaled_data[i : i + context_length])  # Last 30 days
        y.append(
            scaled_data[i + context_length : i + context_length + prediction_length, 0]
        )  # Predict next 3 days close price

    return np.array(X), np.array(y), scaler


def fine_tune_model():
    """
    Fine-tunes the pre-trained stock prediction model with multiple stocks.
    """
    model = load_pretrained_model()
    stock_symbols = ["AAPL", "MSFT", "TSLA", "GOOGL"]  # ✅ Training on 4 stocks

    X_train_list, y_train_list = [], []
    scalers = {}

    for symbol in stock_symbols:
        X_train, y_train, scaler = prepare_stock_data_for_transformer(symbol)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        scalers[symbol] = scaler  # ✅ Save scaler for later

    # Stack all stock data
    X_train_combined = np.vstack(X_train_list)
    y_train_combined = np.vstack(y_train_list)

    # Convert data to PyTorch tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_combined, dtype=torch.float32),
        torch.tensor(y_train_combined, dtype=torch.float32),
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=10,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("fine_tuned_stock_model")
    np.save("scalers.npy", scalers)  # ✅ Save scalers for future use

    return model, scalers


def predict_next_3_days(symbol):
    """
    Uses the fine-tuned model to predict stock prices for the next 3 days.
    """
    model, scaler = fine_tune_model(symbol)  # Train the model if not already fine-tuned
    recent_stock_data = StockService.merge_sentiment_with_stock(symbol).tail(
        30
    )  # Get last 30 days data

    # Scale data
    features = [
        "close",
        "SMA_20",
        "EMA_20",
        "RSI",
        "MACD",
        "BB_Upper",
        "BB_Lower",
        "Sentiment",
    ]
    recent_stock_data_scaled = scaler.transform(recent_stock_data[features])

    # Convert to tensor and predict
    input_tensor = torch.tensor([recent_stock_data_scaled], dtype=torch.float32)
    predictions = model.generate(input_tensor)  # Generate 3-day forecast

    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(predictions.detach().numpy())

    return predictions


#  Run the scraper inside Flask app context
if __name__ == "__main__":
    with app.app_context():  # Ensure MongoDB is initialized before running
        fine_tune_model()
