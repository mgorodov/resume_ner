from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from app.core.security import verify_token
from app.database.session import get_db
from app.core.config import settings

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    return username


def get_current_admin(
    username: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from app.database.crud import get_user_by_username
    
    user = get_user_by_username(db, username)
    if not user or not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return user


def verify_delete_token(delete_token: Optional[str] = Header(None)):
    """Verify delete confirmation token"""
    if not delete_token or delete_token != "CONFIRM_DELETE_12345":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Delete confirmation token required"
        )
    return True