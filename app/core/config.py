
from dotenv import load_dotenv
import os
import stripe

load_dotenv(".env")

# Database
DB_USERNAME = os.getenv("usern")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_PORT = os.getenv("port")
DB_NAME = os.getenv("database")
DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

# CORS / Frontend
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
stripe.api_key = STRIPE_SECRET_KEY

# DigitalOcean Spaces (S3-compatible)
SPACES_KEY = os.getenv("SPACES_KEY")
SPACES_SECRET = os.getenv("SPACES_SECRET")
SPACES_REGION = os.getenv("SPACES_REGION", "ams3")
SPACES_ENDPOINT = (os.getenv("SPACES_ENDPOINT", "https://ams3.digitaloceanspaces.com")).rstrip("/")
SPACES_BUCKET = os.getenv("SPACES_BUCKET")
SPACES_FOLDER = (os.getenv("SPACES_FOLDER") or "").strip("/")
SPACES_CDN_BASE = (os.getenv("SPACES_CDN_BASE") or "").rstrip("/")

# Stock reservations
RES_HOLD_MINUTES = int(os.getenv("RES_HOLD_MINUTES", "15"))

#Login
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "60"))
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() == "true"
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN") or None