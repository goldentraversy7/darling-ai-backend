import os

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
JWT_SECRET_KEY = SECRET_KEY
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/mydatabase")
