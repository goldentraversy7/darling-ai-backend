from app import create_app
import schedule
import time
from app.ai.rl_update import update_model_with_actual_data

# Initialize Flask app
app = create_app()


def daily_update():
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    for symbol in symbols:
        update_model_with_actual_data(symbol)


# Schedule RL updates every 24 hours
schedule.every().day.at("16:30").do(daily_update)  # Market close time

if __name__ == "__main__":
    with app.app_context():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Wait 1 minute before checking again
