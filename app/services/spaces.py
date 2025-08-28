
import uuid
import mimetypes

import boto3
from botocore.client import Config

from app.core.config import (
    SPACES_KEY, SPACES_SECRET, SPACES_REGION, SPACES_ENDPOINT, SPACES_BUCKET,
    SPACES_FOLDER, SPACES_CDN_BASE
)
from app.utils.slugify import slugify

_boto_session = boto3.session.Session()
s3 = _boto_session.client(
    "s3",
    region_name=SPACES_REGION,
    endpoint_url=SPACES_ENDPOINT,
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET,
    config=Config(signature_version="s3v4"),
)

def guess_content_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"

def spaces_key_for(name: str, original_filename: str) -> str:
    base = slugify(name) or "image"
    ext = (("." + original_filename.split(".")[-1]) if "." in original_filename else ".jpg").lower()
    fname = f"{base}-{uuid.uuid4().hex}{ext}"
    return "/".join(x for x in [SPACES_FOLDER, fname] if x)

def public_url_for(key: str) -> str:
    if SPACES_CDN_BASE:
        return f"{SPACES_CDN_BASE}/{key}"
    return f"{SPACES_ENDPOINT}/{SPACES_BUCKET}/{key}"

def upload_to_spaces(file_bytes: bytes, key: str, content_type: str):
    s3.put_object(
        Bucket=SPACES_BUCKET,
        Key=key,
        Body=file_bytes,
        ACL="public-read",
        ContentType=content_type,
        CacheControl="public, max-age=31536000, immutable",
    )
