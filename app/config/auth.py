from pathlib import Path

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent


class AuthJWTConfig(BaseModel):
    private_key_path: Path = PROJECT_ROOT / "certs" / "jwt-private.pem"
    public_key_path: Path = PROJECT_ROOT / "certs" / "jwt-public.pem"
    algorithm: str = "RS256"
    access_token_expire_minutes: int = 60 * 24
