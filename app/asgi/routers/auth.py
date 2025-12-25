from typing import Annotated
from fastapi import APIRouter, HTTPException

from fastapi.security import OAuth2PasswordRequestForm
import sqlalchemy as sa
from app.asgi.auth.deps import require_oauth, UserInfo
from app.asgi.auth.utils import encode_jwt, validate_password
from app.config.settings import settings
from app.asgi.state import DBSessionDep
from app.db.models import UserOrm
from fastapi import Depends, status
import uuid
import datetime as dt

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/insecure")
async def check_insecure():
    return {"msg": "success"}


@router.get("/secure/user")
async def check_user_role(user_info: Annotated[UserInfo, Depends(require_oauth("user"))]):
    return {"msg": "success", "user_info": user_info}


@router.get("/secure/admin")
async def check_admin_role(user_info: Annotated[UserInfo, Depends(require_oauth("admin"))]):
    return {"msg": "success", "user_info": user_info}


@router.post("/login")
async def auth_issue_jwt(
    db_session: DBSessionDep,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    sql = sa.select(UserOrm).where(UserOrm.username == form_data.username)
    query_result = await db_session.execute(sql)
    user_orm = query_result.scalar_one_or_none()
    if user_orm is None or not validate_password(password=form_data.password, hashed_password=user_orm.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    now = dt.datetime.now(dt.timezone.utc)
    expire = now + dt.timedelta(minutes=settings.auth_jwt.access_token_expire_minutes)
    jwt_payload = {
        "sub": user_orm.username,  # subject
        "roles": user_orm.roles,
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
    }

    token = encode_jwt(jwt_payload)
    return {"access_token": token, "token_type": "Bearer"}
