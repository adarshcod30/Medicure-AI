"""
Password hashing and JWT primitives.

`bcrypt` is called directly rather than through passlib: passlib 1.7.4
inspects `bcrypt.__about__`, which bcrypt 4.x removed, producing a spurious
"error reading bcrypt version" on every import. One indirection fewer, one
confusing log line fewer.

Tokens are deliberately minimal — subject, email, expiry. Claims describing
who a user *is* belong in the users collection, where they can be edited
without invalidating every session.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt


def hash_password(password: str) -> str:
    # bcrypt silently truncates beyond 72 bytes; refuse instead of truncating.
    if len(password.encode("utf-8")) > 72:
        raise ValueError("password longer than 72 bytes")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        # Malformed stored hash. Treat as "does not match", not a 500 —
        # the caller cannot fix it and must not learn which failure it was.
        return False


def create_token(*, subject: str, email: str, secret: str, algorithm: str, expiry_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": subject,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expiry_minutes)).timestamp()),
    }
    return jwt.encode(claims, secret, algorithm=algorithm)


def decode_token(token: str, *, secret: str, algorithm: str) -> dict | None:
    """Valid claims, or None. Expiry is enforced by the library."""
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError:
        return None
