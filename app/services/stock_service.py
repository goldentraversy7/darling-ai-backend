import os
import requests
from requests.auth import HTTPBasicAuth

class StockService:
    @staticmethod
    def get_stock_data():
        """
        Fetch order history from Schwab API.
        """
        try:
            return "ok"
        except requests.exceptions.RequestException as e:
            print(f"Error fetching order history: {e}")
            return {"error": str(e)}
