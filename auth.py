"""
Authentication utilities: API Key and JWT support for FastAPI endpoints.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings


security = HTTPBearer(auto_error=False)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(tz=timezone.utc) + (expires_delta or timedelta(hours=settings.JWT_EXPIRATION_HOURS))
    payload = {"sub": subject, "exp": expire}
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token


def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """Authenticate via API key or Bearer JWT.

    Returns a username/subject string. In non-production, if neither is provided,
    returns "anonymous" to simplify local development.
    """

    # API key path
    if x_api_key:
        if x_api_key == settings.API_KEY:
            return "api-key-user"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    # Bearer token path
    if credentials and credentials.scheme.lower() == "bearer":
        payload = verify_token(credentials.credentials)
        return payload.get("sub", "user")

    # Allow anonymous in non-production
    if settings.ENVIRONMENT != "production":
        return "anonymous"

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization required")

