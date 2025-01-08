from flask_jwt_extended import create_access_token
from flask_bcrypt import check_password_hash
from datetime import timedelta
from app.models import User

class AuthService:
    @staticmethod
    def register_user(data):
        if User.find_user_by_email(data["email"]):
            return {"message": "Email already registered", "status": 400}
        return User.create_user(data)

    @staticmethod
    def login_user(data):
        user = User.find_user_by_email(data["email"])
        if user and check_password_hash(user["password"], data["password"]):
            token = create_access_token(
                identity={"email": user["email"]}, expires_delta=timedelta(hours=1)
            )
            return {"message": "Login successful", "token": token, "status": 200}
        return {"message": "Invalid credentials", "status": 401}
