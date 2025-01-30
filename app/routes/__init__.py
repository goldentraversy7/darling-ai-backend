from app.routes.auth_routes import auth_routes  # Import your auth blueprint
from app.routes.stock_routes import stock_routes  # Import your stock blueprint
from app.routes.background_routes import background_routes  # Import your background_routes

# Initialize a list of blueprints
blueprints = [
    auth_routes,  # Add other blueprints here as you create them
    stock_routes,
    background_routes,
]
