import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from flask import current_app
from app.ai.stock_lstm import StockLSTM
from app.ai.preprocess import prepare_stock_data


def get_app():
    """Get the Flask app context safely to avoid circular import."""
    from app import (
        create_app,
    )  # Moved import inside function to prevent circular import

    return create_app()


# Detect device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to stored model
MODEL_DIR = os.path.abspath("models_storage")
MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm_model.pth")


def fine_tune_lstm(stock_symbols, num_epochs=40, patience=10):
    """
    Fine-tunes the LSTM model with stock and sentiment data.
    Implements validation loss tracking and Early Stopping.
    """

    app = get_app()
    with app.app_context():  # ✅ Ensures Flask app context is active
        X, y = prepare_stock_data(stock_symbols)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Split dataset (80% Train, 20% Validation)
    total_samples = len(X_tensor)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_dataset, val_dataset = random_split(
        TensorDataset(X_tensor, y_tensor), [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    input_size = X.shape[2]
    model = StockLSTM(input_size)

    # Load existing model if available
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    # Early stopping parameters
    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(num_epochs):
        train_loss, val_loss = 0.0, 0.0

        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    return model


# Run inside Flask app context
if __name__ == "__main__":
    app = get_app()  # Initialize Flask app safely
    with app.app_context():
        fine_tune_lstm(["AAPL", "GOOGL", "MSFT", "TSLA"])  # Train on multiple stocks
