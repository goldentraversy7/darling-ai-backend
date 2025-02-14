import numpy as np


def calculate_reward(predicted_prices, actual_prices):
    """
    Computes reward based on the accuracy of predicted stock prices.

    Reward is calculated as the inverse of Mean Squared Error (MSE) to encourage
    lower error values.

    Arguments:
    - predicted_prices: np.array of shape (3,) → Predicted close prices for 3 days.
    - actual_prices: np.array of shape (3,) → Actual close prices for the same days.

    Returns:
    - reward (float): Higher reward means better prediction accuracy.
    """
    if len(predicted_prices) != len(actual_prices):
        raise ValueError("Predicted and actual prices must have the same length!")

    # Compute Mean Squared Error (MSE)
    mse = np.mean((predicted_prices - actual_prices) ** 2)

    # Convert MSE to a reward (Lower MSE = Higher Reward)
    reward = 1 / (
        1 + mse
    )  # Ensures reward is between (0, 1], avoiding division by zero

    return reward
