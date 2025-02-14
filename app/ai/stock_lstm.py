import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    LSTM model for stock price prediction with Dropout to prevent overfitting.
    """

    def __init__(
        self, input_size, hidden_size=256, num_layers=3, output_size=3, dropout_prob=0.3
    ):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Additional dropout before output layer

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Apply dropout to last time step
        return self.fc(lstm_out)
