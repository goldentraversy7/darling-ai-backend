from app.extensions import mongo
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

class User:
    @staticmethod
    def create_user(data):
        data["password"] = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
        mongo.db.users.insert_one(data)
        return {"message": "User registered successfully", "status": 201}

    @staticmethod
    def find_user_by_email(email):
        return mongo.db.users.find_one({"email": email})
