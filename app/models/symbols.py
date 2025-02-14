from app.extensions import mongo
from datetime import datetime


class Symbol:
    """
    MongoDB Model for tracking stock symbols.
    """

    @staticmethod
    def is_symbol_tracked(symbol):
        """
        Check if the stock symbol already exists in the database.
        """
        existing_symbol = mongo.db.symbols.find_one({"symbol": symbol})
        return existing_symbol is not None

    @staticmethod
    def save_symbol(symbol):
        """
        Save a new stock symbol in the database if it doesn't exist.
        """
        if not Symbol.is_symbol_tracked(symbol):
            symbol_entry = {
                "symbol": symbol,
                "created_at": datetime.utcnow(),
            }
            mongo.db.symbols.insert_one(symbol_entry)
            print(f"New symbol {symbol} saved to MongoDB.")

    @staticmethod
    def fetch_all_symbols():
        """
        Retrieve all tracked stock symbols from the database.
        """
        symbols_cursor = mongo.db.symbols.find({}, {"_id": 0, "symbol": 1})
        return [doc["symbol"] for doc in symbols_cursor]
