"""
Database engine, session factory, and declarative base.
Uses SQLite for the MVP (file-based, zero external dependency, perfect
for a 4-person undergraduate project and local grading/demo).

Why SQLite over Postgres/MySQL for this MVP:
- No server to install/run/manage during development or demo.
- The whole DB is a single file that can be committed as a fixture
  or wiped for a clean demo.
- SQLAlchemy makes it trivial to swap to Postgres later (just change
  DATABASE_URL) without touching the rest of the code.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called once on app startup."""
    from app.models import db_models  # noqa: F401 (register models with Base)
    Base.metadata.create_all(bind=engine)
