from app import create_app
import schedule
import time
import pytz
import datetime
from app.ai.rl_update import update_model_with_actual_data
from app.models import Symbol  # Dynamically fetch symbols

# Initialize Flask app
app = create_app()

# Detect & set timezone (e.g., New York for US markets)
MARKET_TIMEZONE = pytz.timezone("America/New_York")  # Adjust as needed
SCHEDULED_TIME = "18:30"  # Market close time (adjust if necessary)


def get_tracked_symbols():
    """Fetch stock symbols dynamically from MongoDB."""
    with app.app_context():
        return Symbol.fetch_all_symbols()  # Assuming this method fetches all symbols


def daily_update():
    """Run daily model updates using actual stock data."""
    symbols = get_tracked_symbols()
    if not symbols:
        print("⚠️ No tracked symbols found in the database.")
        return

    print(f"🚀 Running RL updates for symbols: {symbols}")
    for symbol in symbols:
        try:
            update_model_with_actual_data(symbol)
            print(f"✅ Model updated for {symbol}")
        except Exception as e:
            print(f"❌ Error updating {symbol}: {e}")


def set_schedule():
    """Schedule the task dynamically based on timezone."""
    local_time = datetime.datetime.now(MARKET_TIMEZONE).strftime("%H:%M")
    print(
        f"⏳ Current Local Time: {local_time} | Scheduling updates at {SCHEDULED_TIME} {MARKET_TIMEZONE}"
    )

    # Schedule the job at market close time
    schedule.every().day.at(SCHEDULED_TIME, MARKET_TIMEZONE).do(daily_update)


if __name__ == "__main__":
    with app.app_context():
        set_schedule()
        print("✅ Scheduler is running...")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check schedule every minute
