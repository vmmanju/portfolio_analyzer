from sqlalchemy.orm import Session
from typing import Generator

from app.database import SessionLocal

def get_db() -> Generator[Session, None, None]:
    """
    Dependency that yields a DB session and ensures it is closed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user():
    """
    Stub for future auth, returns a mock user ID.
    Will be replaced with proper JWT verification later.
    """
    return {"user_id": 1, "username": "admin"}
