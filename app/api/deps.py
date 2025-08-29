
import re
from fastapi import HTTPException, Request, Depends
from app.core.security import decode_token

CART_ID_REGEX = re.compile(r"^[A-Za-z0-9]{32}$")

def require_cart_id(request: Request) -> str:
    cid = request.headers.get("X-Cart-Id") or request.cookies.get("cart_id")
    if not cid or not CART_ID_REGEX.fullmatch(cid):
        raise HTTPException(status_code=400, detail="Missing or invalid cart id (32 alnum chars).")
    return cid

def get_current_user(request: Request):
    token = (request.cookies.get("access_token")
             or (request.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
             or None)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        data = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"email": data.get("sub"), "role": data.get("role")}

def require_admin(user = Depends(get_current_user)):
    if user["role"] != "ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    return user