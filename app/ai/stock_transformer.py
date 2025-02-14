import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerForPrediction


class StockTransformer(nn.Module):
    """
    Transformer model for stock price prediction.
    """

    def __init__(self, input_size, output_size=3):
        super(StockTransformer, self).__init__()
        self.model = TimeSeriesTransformerForPrediction.from_pretrained(
            "huggingface/time-series-transformer-tourism-monthly"
        )

        self.fc = nn.Linear(
            self.model.config.d_model, output_size
        )  # Use correct hidden size

    def forward(self, x):
        """
        Forward pass for the Transformer model.
        """
        batch_size, sequence_length, num_features = x.shape  # Example: (B, T, F)

        # 🔹 **Ensure `past_values` has correct shape**
        past_values = x[:, :, :1]  # Only 'close' price (adjust if needed)

        # 🔹 **Ensure `past_time_features` has correct shape**
        past_time_features = torch.zeros(
            (batch_size, sequence_length, 1), dtype=torch.float32
        )

        # 🔹 **Ensure `past_observed_mask` has correct shape**
        past_observed_mask = torch.ones(
            (batch_size, sequence_length), dtype=torch.float32
        )

        # 🔹 **Pass inputs to the model**
        output = self.model(
            past_values=past_values,  # Must be shape (B, T, 1)
            past_time_features=past_time_features,  # Must be shape (B, T, 1)
            past_observed_mask=past_observed_mask,  # Must be shape (B, T)
        ).logits  # Extract predicted values

        return self.fc(output[:, -1, :])  # Pass through fully connected layer
