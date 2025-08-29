from pydantic import BaseModel, SecretStr
from typing import Literal

UserRole = Literal["ADMIN", "REGULAR"]

class LoginIn(BaseModel):
    username: str
    password: SecretStr

class UserOut(BaseModel):
    username: str
    role: UserRole

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
