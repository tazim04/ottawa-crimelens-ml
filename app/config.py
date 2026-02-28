import os

# Load environment variables from .env file

DB_HOST = os.getenv("NEON_DB_URL")
DB_NAME = os.getenv("NEON_DB_NAME", "neondb")
DB_USER = os.getenv("NEON_DB_USER", "neondb_owner")
DB_PASSWORD = os.getenv("NEON_DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}/{DB_NAME}?sslmode=require"
)