from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services import BackgroundService

# Define the Blueprint
background_routes = Blueprint("background_routes", __name__)


@background_routes.route("/api/background/news", methods=["GET"])
# @jwt_required()
def fetch_background_news():
    # current_user = get_jwt_identity()  # Get the logged-in user's identity
    # if not current_user:
    #     return jsonify({"message": "Unauthorized access", "status": 401}), 401

    # Call StockService to fetch stock data
    query = request.args.get("query")
    result = BackgroundService.fetch_background_news(query=query)
    if "error" in result:
        return (
            jsonify(
                {
                    "message": "Failed to fetch news result",
                    "error": result["error"],
                    "status": 500,
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "message": "Background news retrieved successfully",
                "data": result,
                "status": 200,
            }
        ),
        200,
    )


@background_routes.route("/api/background/social-post", methods=["GET"])
# @jwt_required()
def fetch_socialmedia_posts():
    # current_user = get_jwt_identity()  # Get the logged-in user's identity
    # if not current_user:
    #     return jsonify({"message": "Unauthorized access", "status": 401}), 401

    # Call StockService to fetch stock data
    query = request.args.get("query")
    result = BackgroundService.fetch_socialmedia_posts(query=query)
    if "error" in result:
        return (
            jsonify(
                {
                    "message": "Failed to fetch news result",
                    "error": result["error"],
                    "status": 500,
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "message": "Background news retrieved successfully",
                "data": result,
                "status": 200,
            }
        ),
        200,
    )


@background_routes.route("/api/background/public-record", methods=["GET"])
# @jwt_required()
def fetch_public_records():
    # current_user = get_jwt_identity()  # Get the logged-in user's identity
    # if not current_user:
    #     return jsonify({"message": "Unauthorized access", "status": 401}), 401

    # Call StockService to fetch stock data
    query = request.args.get("query")
    result = BackgroundService.fetch_public_records(query=query)
    if "error" in result:
        return (
            jsonify(
                {
                    "message": "Failed to fetch news result",
                    "error": result["error"],
                    "status": 500,
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "message": "Background news retrieved successfully",
                "data": result,
                "status": 200,
            }
        ),
        200,
    )
