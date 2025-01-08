from flask import Blueprint, request, jsonify
from app.services import AuthService

# Define the Blueprint
auth_routes = Blueprint("auth_routes", __name__)

@auth_routes.route("/api/register", methods=["POST"])
def register():
    data = request.json
    response = AuthService.register_user(data)
    return jsonify(response), response["status"]

@auth_routes.route("/api/login", methods=["POST"])
def login():
    data = request.json
    response = AuthService.login_user(data)
    return jsonify(response), response["status"]
