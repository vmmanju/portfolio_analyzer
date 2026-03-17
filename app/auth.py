"""Authentication helpers for multi-user support.

Uses Streamlit's native OIDC auth (v1.42+) with Google provider.
The `require_login()` function should be called at the top of the dashboard
to gate access. It returns the DB User object for the logged-in user.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.database import get_db_context
from app.models import User


def get_or_create_user(
    email: str,
    name: Optional[str] = None,
    picture_url: Optional[str] = None,
    skip_update: bool = False,
) -> User:
    """Find existing user by email or create a new one.

    On login (when skip_update=False), updates last_login timestamp and profile fields.
    Returns a lightweight UserInfo object.
    """
    with get_db_context() as db:
        user = db.execute(
            select(User).where(User.email == email)
        ).scalar_one_or_none()

        if user is None:
            user = User(
                email=email,
                name=name,
                picture_url=picture_url,
            )
            db.add(user)
            db.flush()  # Assign ID
        elif not skip_update:
            # Update profile only if requested (avoid redundant writes on every rerun)
            user.last_login = datetime.utcnow()
            if name:
                user.name = name
            if picture_url:
                user.picture_url = picture_url

        user_id = user.id
        user_email = user.email
        user_name = user.name
        user_picture = user.picture_url

    # Return a lightweight dict-like namespace (avoid detached ORM issues)
    class UserInfo:
        def __init__(self, id, email, name, picture_url):
            self.id = id
            self.email = email
            self.name = name
            self.picture_url = picture_url
        def __repr__(self):
            return f"<UserInfo(id={self.id}, email={self.email!r})>"

    return UserInfo(user_id, user_email, user_name, user_picture)


def require_login():
    """Gate function for Streamlit dashboard.

    Call at the top of main(). Returns UserInfo if logged in, or
    shows login button and calls st.stop() to halt rendering.
    """
    import streamlit as st

    # Check if user is logged in via Streamlit OIDC
    if not st.user or not getattr(st.user, "is_logged_in", False):
        st.title("🔐 AI Stock Engine")
        st.markdown("Please sign in with your Google account to continue.")
        st.login("google")
        st.stop()

    # User is logged in — get or create DB record
    email = st.user.email
    name = getattr(st.user, "name", None) or getattr(st.user, "given_name", None)
    picture = getattr(st.user, "picture", None)

    # Performance: only update DB timestamps once per session
    skip_upd = st.session_state.get("auth_updated", False)
    user_info = get_or_create_user(email, name, picture, skip_update=skip_upd)
    
    if not skip_upd:
        st.session_state["auth_updated"] = True
        
    return user_info
