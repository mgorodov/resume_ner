from typing import Annotated, Callable
from jwt.exceptions import InvalidTokenError
from fastapi import (
    Depends,
    HTTPException,
    status,
)
from fastapi.security import OAuth2PasswordBearer

from app.asgi.auth.utils import decode_jwt
from pydantic import BaseModel


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login/")


class UserInfo(BaseModel):
    actor: str
    roles: list[str]
    jwt_payload: dict

    def verify_actor(self, actor: str):
        if self.actor != actor:
            self.verify_role("admin")

    def verify_role(self, required_role: str):
        if required_role not in self.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"You don't have `{required_role}` role")


def require_oauth(required_role: str = "user") -> Callable[..., UserInfo]:
    def inner(token: Annotated[str, Depends(oauth2_scheme)]) -> UserInfo:
        try:
            jwt_payload = decode_jwt(token=token)
        except InvalidTokenError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        token_info = UserInfo(actor=jwt_payload["sub"], roles=jwt_payload["roles"], jwt_payload=jwt_payload)
        token_info.verify_role(required_role)
        return token_info

    return inner


UserRoleDep = Annotated[UserInfo, Depends(require_oauth("user"))]
AdminRoleDep = Annotated[UserInfo, Depends(require_oauth("admin"))]
