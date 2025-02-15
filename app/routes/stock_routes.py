from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services import StockService
from app.models import Symbol  # Import MongoDB model
from app.ai.predict_lstm import check_and_train_new_symbol
import yfinance as yf


# Define the Blueprint
stock_routes = Blueprint("stock_routes", __name__)


@stock_routes.route("/api/stock/", methods=["GET"])
# @jwt_required()
def get_stock_data():
    # current_user = get_jwt_identity()  # Get the logged-in user's identity
    # if not current_user:
    #     return jsonify({"message": "Unauthorized access", "status": 401}), 401

    # Call StockService to fetch stock data
    symbol = request.args.get("symbol")
    company = request.args.get("company")

    # ✅ Check if symbol is provided
    if not symbol:
        return jsonify({"message": "Stock symbol is required", "status": 400}), 400

    # ✅ Validate the stock symbol using Yahoo Finance before proceeding
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        if "symbol" not in stock_info:
            return (
                jsonify({"message": f"Invalid stock symbol: {symbol}", "status": 400}),
                400,
            )
    except Exception as e:
        return (
            jsonify(
                {
                    "message": f"Error validating symbol: {symbol}",
                    "error": str(e),
                    "status": 500,
                }
            ),
            500,
        )

    if not Symbol.is_symbol_tracked(symbol):
        Symbol.save_symbol(symbol)  # Save new symbol
        train_flag = check_and_train_new_symbol(
            symbol
        )  # Start fine-tuning in the background
        if train_flag:
            return (
                jsonify(
                    {
                        "message": f"New stock {symbol} detected. Training model...",
                        "data": {},
                        "status": 202,
                    }
                ),
                202,
            )

    stock_data = StockService.fetch_stock_data(symbol=symbol, company=company)
    if "error" in stock_data:
        return (
            jsonify(
                {
                    "message": "Failed to fetch stock data",
                    "error": stock_data["error"],
                    "status": 500,
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "message": "Stock data retrieved successfully",
                "data": stock_data,
                "status": 200,
            }
        ),
        200,
    )
