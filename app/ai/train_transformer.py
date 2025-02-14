from app import create_app
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from app.ai.stock_transformer import StockTransformer
from app.ai.preprocess import prepare_stock_data

# Initialize Flask app
app = create_app()


def fine_tune_transformer():
    """
    Fine-tunes a single Transformer model for multiple stocks.
    """
    print("Fine-tuning Transformer model for multiple stocks...")

    # Define stock symbols to train on
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]  # Modify as needed

    # Load training data for multiple stocks
    X_train, y_train = prepare_stock_data(stock_symbols)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize Transformer model
    input_size = X_train.shape[2]  # Number of features
    model = StockTransformer(input_size)

    # Loss function & optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}"
        )

    # Save fine-tuned model
    torch.save(model.state_dict(), "models_storage/fine_tuned_transformer.pth")
    print("Model saved: models_storage/fine_tuned_transformer.pth")

    return model


# Run training if executed directly
if __name__ == "__main__":
    with app.app_context():
        fine_tune_transformer()
