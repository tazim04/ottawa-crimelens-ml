from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from app.config import DATABASE_URL

# Create a SQLAlchemy engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    future=True,
)