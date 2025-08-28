
import re
from fastapi import HTTPException, Request

CART_ID_REGEX = re.compile(r"^[A-Za-z0-9]{32}$")

def require_cart_id(request: Request) -> str:
    cid = request.headers.get("X-Cart-Id") or request.cookies.get("cart_id")
    if not cid or not CART_ID_REGEX.fullmatch(cid):
        raise HTTPException(status_code=400, detail="Missing or invalid cart id (32 alnum chars).")
    return cid
