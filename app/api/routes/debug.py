
from fastapi import APIRouter
from app.services.spaces import s3
from app.core.config import (
    SPACES_REGION, SPACES_ENDPOINT, SPACES_BUCKET, SPACES_FOLDER, SPACES_CDN_BASE,
    SPACES_KEY, SPACES_SECRET
)

router = APIRouter()

@router.get("/debug/spaces/health")
def spaces_health():
    cfg = {
        "SPACES_REGION": SPACES_REGION,
        "SPACES_ENDPOINT": SPACES_ENDPOINT,
        "SPACES_BUCKET": SPACES_BUCKET,
        "SPACES_FOLDER": SPACES_FOLDER,
        "SPACES_CDN_BASE": SPACES_CDN_BASE,
        "SPACES_KEY_set": bool(SPACES_KEY),
        "SPACES_SECRET_set": bool(SPACES_SECRET),
    }
    try:
        s3.head_bucket(Bucket=SPACES_BUCKET)
        status = "ok: head_bucket passed"
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"
    return {"status": status, "config": cfg}
