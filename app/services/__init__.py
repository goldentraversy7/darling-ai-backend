from app.services.auth_service import AuthService
from app.services.stock_service import StockService

# Expose the services for easier import
__all__ = ["AuthService", StockService]
