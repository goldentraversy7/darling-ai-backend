from flask import Flask
from flask_cors import CORS
from app.extensions import mongo, jwt
from app.routes import blueprints  # Import blueprints from routes/__init__.py


def create_app():
    app = Flask(__name__)

    # Load configurations
    app.config.from_pyfile("config.py")

    # Initialize extensions
    mongo.init_app(app)
    jwt.init_app(app)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

    # Register blueprints
    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    return app
