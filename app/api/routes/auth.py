from fastapi import APIRouter, Depends, HTTPException, Response, Request
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import LoginIn, UserOut, TokenOut
from app.core.security import verify_password, create_access_token, decode_token
from app.core.config import COOKIE_SECURE, COOKIE_DOMAIN

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "access_token"

def set_login_cookie(response: Response, token: str):
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        domain=COOKIE_DOMAIN,
        path="/",
        max_age=60 * 60 * 24 * 7,
    )

def clear_login_cookie(response: Response):
    response.delete_cookie(COOKIE_NAME, domain=COOKIE_DOMAIN, path="/")

@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, response: Response, db: Session = Depends(get_db)):
    user = db.get(User, body.username)
    if not user or not verify_password(body.password.get_secret_value(), user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(sub=user.username, role=user.role)
    set_login_cookie(response, token)
    return TokenOut(access_token=token, user=UserOut(username=user.username, role=user.role))

@router.post("/logout", status_code=204)
def logout(response: Response):
    clear_login_cookie(response)
    return

@router.get("/me", response_model=UserOut)
def me(request: Request) -> UserOut:
    token = (request.cookies.get(COOKIE_NAME)
             or (request.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
             or None)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        data = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    return UserOut(username=data.get("sub"), role=data.get("role"))
