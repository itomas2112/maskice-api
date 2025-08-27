# main.py
import os, uuid, mimetypes
from typing import List, Literal, Optional
import re
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request, Query, Path, UploadFile, File, Form, Body, Header  # ← add UploadFile, File, Form
from sqlalchemy.exc import IntegrityError

from sqlalchemy import select as sa_select

from enum import Enum as PyEnum


import boto3  # ← add
from botocore.client import Config  # ← add

from fastapi import FastAPI, HTTPException, Request, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint
from dotenv import load_dotenv
from pydantic import EmailStr  # top of file

from sqlalchemy import (
    create_engine, Text, Integer, DateTime, select, func as sa_func, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship, selectinload, aliased
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import String, cast, Index, Enum, and_, or_, update

import stripe

# ---------- Config ----------
load_dotenv(r'.env')

username = os.getenv("usern")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")

SPACES_KEY = os.getenv("SPACES_KEY")
SPACES_SECRET = os.getenv("SPACES_SECRET")
SPACES_REGION = os.getenv("SPACES_REGION", "ams3")
SPACES_ENDPOINT = os.getenv("SPACES_ENDPOINT", "https://ams3.digitaloceanspaces.com").rstrip("/")
SPACES_BUCKET = os.getenv("SPACES_BUCKET")
SPACES_FOLDER = os.getenv("SPACES_FOLDER", "").strip("/")
SPACES_CDN_BASE = (os.getenv("SPACES_CDN_BASE") or "").rstrip("/")

connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}?sslmode=require'

DATABASE_URL = f'postgresql://{username}:{password}@{host}:{port}/{database}?sslmode=require'

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

_boto_session = boto3.session.Session()
s3 = _boto_session.client(
    "s3",
    region_name=SPACES_REGION,
    endpoint_url=SPACES_ENDPOINT,              # DO Spaces = S3-compatible
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET,
    config=Config(signature_version="s3v4"),
)

# ---------- ORM ----------
class Base(DeclarativeBase):
    pass

class ProductVar(Base):
    __tablename__ = "products"

    # PK
    id: Mapped[str]      = mapped_column(Text, primary_key=True)
    colors: Mapped[str]  = mapped_column(Text, primary_key=True)
    compat: Mapped[str]  = mapped_column(Text, primary_key=True)

    # Attributes
    name: Mapped[str]        = mapped_column(Text, nullable=False)
    image: Mapped[str]       = mapped_column(Text, nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str | None]  = mapped_column(Text, nullable=True)   # e.g. "Case"
    phone: Mapped[str | None] = mapped_column(Text, nullable=True)   # e.g. "iPhone"
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(Text, default="PENDING")
    currency: Mapped[str] = mapped_column(Text, default="EUR")
    subtotal_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    shipping_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    total_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())

    # NEW
    complete: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    items: Mapped[List["OrderItem"]] = relationship(back_populates="order", cascade="all, delete-orphan")
    stripe_session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    payment_intent_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    customer_first_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    customer_last_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    customer_email: Mapped[str | None] = mapped_column(Text, nullable=True)

    ship_address_line1: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_address_line2: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_city: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_postal_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_country: Mapped[str | None] = mapped_column(Text, nullable=True)

class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(Text, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)

    # denormalized snapshot for audit integrity
    product_id: Mapped[str] = mapped_column(Text, nullable=False)
    product_name: Mapped[str] = mapped_column(Text, nullable=False)
    image: Mapped[str] = mapped_column(Text, nullable=False)

    color: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)  # "iPhone 16" | "iPhone 16 Pro"
    qty: Mapped[int] = mapped_column(Integer, nullable=False)

    unit_price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    line_total_cents: Mapped[int] = mapped_column(Integer, nullable=False)

    order: Mapped[Order] = relationship(back_populates="items")

class CartItem(Base):
    __tablename__ = "cart_items"

    cart_id:    Mapped[str] = mapped_column(Text, primary_key=True)   # 32-char cookie/header
    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    color:      Mapped[str] = mapped_column(Text, primary_key=True)
    model:      Mapped[str] = mapped_column(Text, primary_key=True)   # CompatType values
    qty:        Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now(), onupdate=sa_func.now())


# ---------- Lookup tables ----------
class TypeRow(Base):
    __tablename__ = "types"
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())

class PhoneRow(Base):
    __tablename__ = "phones"
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())

class SubphoneRow(Base):
    __tablename__ = "subphones"
    # composite PK (phone, name)
    phone: Mapped[str] = mapped_column(Text, ForeignKey("phones.name", onupdate="CASCADE", ondelete="CASCADE"), primary_key=True)
    name:  Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())


# ---------- Types ----------
CompatType = Literal["iPhone 16", "iPhone 16 Pro"]

class ProductOut(BaseModel):
    id: str
    name: str
    image: str
    colors: List[str]
    compat: List[CompatType]
    price_cents: int
    class Config: from_attributes = True

class CartItemIn(BaseModel):
    product_id: str
    qty: conint(ge=1, le=100)
    color: str
    model: CompatType

class QuoteItemOut(BaseModel):
    product_id: str
    name: str
    color: str
    model: CompatType
    qty: int
    unit_price_cents: int
    line_total_cents: int

class QuoteOut(BaseModel):
    items: List[QuoteItemOut]
    subtotal_cents: int
    shipping_cents: int
    total_cents: int

class OrderItemOut(BaseModel):
    id: str
    product_id: str
    product_name: str
    image: str
    color: str
    model: CompatType
    qty: int
    unit_price_cents: int
    line_total_cents: int
    class Config: from_attributes = True

class OrderListItemOut(BaseModel):
    id: str
    status: str
    currency: str
    subtotal_cents: int
    shipping_cents: int
    total_cents: int
    created_at: str | None = None
    complete: int                    # ← add this
    items: List[OrderItemOut]
    class Config: from_attributes = True

class OrderOut(BaseModel):
    id: str
    status: str
    currency: str
    subtotal_cents: int
    shipping_cents: int
    total_cents: int
    items: List[OrderItemOut]
    class Config: from_attributes = True

class AddressIn(BaseModel):
    line1: str
    line2: str | None = None
    city: str
    postal_code: str
    country: str

class CustomerIn(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    address: AddressIn

class CheckoutPayload(BaseModel):
    items: List[CartItemIn]
    customer: CustomerIn

class ProductVariantOut(BaseModel):
    product_id: str
    colors: str
    image: str
    quantity: int

class ProductRowIn(BaseModel):
    id: str | None = None
    name: str
    image: str
    colors: str
    compat: CompatType
    price_cents: conint(ge=0)
    type: str
    phone: str

class ProductByCompatOut(BaseModel):
    id: str
    name: str
    compat: CompatType
    price_cents: int
    variants: List[ProductVariantOut]
    type: str | None = None
    phone: str | None = None
    # optional: sum of variant quantities for this (name, compat) group
    total_quantity: int | None = None

class CartItemPatch(BaseModel):
    action: Literal["add", "set", "remove"]
    product_id: str
    color: str
    model: CompatType
    qty: conint(ge=0, le=100) = 0  # for add/set; 0 with "set" removes

class CartOut(BaseModel):
    items: List[QuoteItemOut]
    subtotal_cents: int
    shipping_cents: int
    total_cents: int

class TypeOut(BaseModel):
    name: str
    class Config: from_attributes = True

class PhoneOut(BaseModel):
    name: str
    class Config: from_attributes = True

class SubphoneOut(BaseModel):
    phone: str
    name: str
    class Config: from_attributes = True

class TypeIn(BaseModel):
    name: str

class PhoneIn(BaseModel):
    name: str

class SubphoneIn(BaseModel):
    phone: str
    name: str

class ReservationState(str, PyEnum):
    HELD = "HELD"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"

class StockReservation(Base):
    __tablename__ = "stock_reservations"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(Text, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)

    product_id: Mapped[str] = mapped_column(Text, nullable=False)
    color: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)  # compat value
    qty: Mapped[int] = mapped_column(Integer, nullable=False)

    state: Mapped[str] = mapped_column(Text, nullable=False, default="HELD", server_default="HELD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

app = FastAPI(title="Maskino API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#%% Helpers
# ---------- Helpers ----------
def compute_shipping(subtotal_cents: int) -> int:
    # Same rule as UI: free if subtotal >= 25€ or subtotal == 0, else 2€
    return 0 if (subtotal_cents == 0 or subtotal_cents >= 2500) else 200

# Return a product snapshot (incl. image) from validation:
def validate_and_price(items: List[CartItemIn]):
    if not items:
        return [], 0, {}

    wanted_ids = list({i.product_id for i in items})
    with SessionLocal() as db:
        variants = db.scalars(select(ProductVar).where(ProductVar.id.in_(wanted_ids))).all()

    # Build per-product index
    prod_index: dict[str, dict] = {}
    for v in variants:
        entry = prod_index.setdefault(v.id, {
            "name": v.name,
            "price_cents": v.price_cents,
            "colors": set(),
            "compat": set(),
            # map (color, compat) -> image (fallbacks if needed)
            "img_by_variant": {},
            "img_by_color": {},
            "any_image": v.image,
        })
        entry["colors"].add(v.colors)
        entry["compat"].add(v.compat)
        entry["img_by_variant"][(v.colors, v.compat)] = v.image
        entry["img_by_color"].setdefault(v.colors, v.image)
        # Keep first seen as "any_image" in case we need a fallback

    out_items: List[QuoteItemOut] = []
    subtotal = 0

    # snapshot used later when persisting order items / building Stripe line items
    prod_snap: dict[str, dict] = {}

    for it in items:
        entry = prod_index.get(it.product_id)
        if not entry:
            raise HTTPException(status_code=400, detail=f"Unknown product_id '{it.product_id}'")

        if it.color not in entry["colors"]:
            raise HTTPException(status_code=400, detail=f"Invalid color '{it.color}' for product '{it.product_id}'")

        if it.model not in entry["compat"]:
            raise HTTPException(status_code=400, detail=f"Model '{it.model}' not compatible with product '{it.product_id}'")

        unit = entry["price_cents"]
        line_total = unit * it.qty
        subtotal += line_total

        # pick the best image for this variant
        image = (
            entry["img_by_variant"].get((it.color, it.model)) or
            entry["img_by_color"].get(it.color) or
            entry["any_image"]
        )

        # Build/refresh snapshot per product id (used later in create_order/checkout)
        prod_snap[it.product_id] = {
            "id": it.product_id,
            "name": entry["name"],
            # Store a representative image, but we'll pass the per-line chosen image to OrderItem
            "image": image,  # optional; not strictly needed if you always set per-line image below
            "colors": list(entry["colors"]),
            "compat": list(entry["compat"]),
            "price_cents": unit,
        }

        out_items.append(QuoteItemOut(
            product_id=it.product_id,
            name=entry["name"],
            color=it.color,
            model=it.model,
            qty=it.qty,
            unit_price_cents=unit,
            line_total_cents=line_total,
        ))

    return out_items, subtotal, prod_snap, prod_index  # NOTE: return prod_index so we can use images


def _as_list(v: str | list[str] | None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    # split on commas if present; trim whitespace
    parts = [p.strip() for p in v.split(",")]
    return [p for p in parts if p] if len(parts) > 1 else [v.strip()]

def _uniq(seq: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')

def _guess_content_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"

def _spaces_key_for(name: str, original_filename: str) -> str:
    # safe slug + random suffix + original extension
    base = _slugify(name) or "image"
    ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
    fname = f"{base}-{uuid.uuid4().hex}{ext}"
    return "/".join(x for x in [SPACES_FOLDER, fname] if x)

def _public_url_for(key: str) -> str:
    if SPACES_CDN_BASE:
        return f"{SPACES_CDN_BASE}/{key}"
    # default DO endpoint-style URL
    return f"{SPACES_ENDPOINT}/{SPACES_BUCKET}/{key}"

def _upload_to_spaces(file_bytes: bytes, key: str, content_type: str):
    s3.put_object(
        Bucket=SPACES_BUCKET,
        Key=key,
        Body=file_bytes,
        ACL="public-read",
        ContentType=content_type,
        CacheControl="public, max-age=31536000, immutable",
    )

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

RES_HOLD_MINUTES = int(os.getenv("RES_HOLD_MINUTES", "15"))

Index("ix_reservations_active", StockReservation.product_id, StockReservation.color, StockReservation.model, StockReservation.state, StockReservation.expires_at)

def release_expired_reservations(db):
    """Return stock for any HELD reservations past expires_at, then mark them RELEASED.
       Do this inside a transaction before new reservations/quotes."""
    # Find all expired HELD reservations
    expired = db.scalars(
        sa_select(StockReservation)
        .where(
            StockReservation.state == "HELD",
            StockReservation.expires_at <= now_utc()
        )
        .with_for_update(skip_locked=True)
    ).all()
    if not expired:
        return

    # Aggregate to minimize UPDATEs per ProductVar row
    by_key: dict[tuple[str,str,str], int] = {}
    for r in expired:
        by_key[(r.product_id, r.color, r.model)] = by_key.get((r.product_id, r.color, r.model), 0) + r.qty

    # Return stock
    for (pid, color, model), qty in by_key.items():
        pv = db.get(ProductVar, {"id": pid, "colors": color, "compat": model})
        if pv:
            # lock the row
            db.execute(
                sa_select(ProductVar)
                .where(
                    ProductVar.id == pid,
                    ProductVar.colors == color,
                    ProductVar.compat == model
                )
                .with_for_update()
            )
            pv.quantity = pv.quantity + qty

    # Mark reservations RELEASED
    for r in expired:
        r.state = "RELEASED"
    # Commit happens in the caller's transaction

def ensure_user_reservation_limits(db, email: str | None, ip: str | None, max_active_items: int = 10):
    # Count active HELD reservations tied to pending orders for this email or IP
    q = sa_select(sa_func.coalesce(sa_func.sum(StockReservation.qty), 0)).join(Order,
                                                                               Order.id == StockReservation.order_id).where(
        StockReservation.state == "HELD",
        Order.status == "PENDING",
        or_(
            and_(email is not None, Order.customer_email == email),
            and_(ip is not None, Order.ship_address_line1 == None, True)
            # replace with your own IP storage if you keep it
        )
    )
    active_qty = db.scalar(q) or 0
    if active_qty > max_active_items:
        raise HTTPException(status_code=429, detail="Too many active holds. Complete or wait for holds to expire.")

CART_ID_REGEX = re.compile(r"^[A-Za-z0-9]{32}$")

def _require_cart_id(request: Request) -> str:
    # Prefer header, fallback to cookie
    cid = request.headers.get("X-Cart-Id") or request.cookies.get("cart_id")
    if not cid or not CART_ID_REGEX.fullmatch(cid):
        raise HTTPException(status_code=400, detail="Missing or invalid cart id (32 alnum chars).")
    return cid

def _cart_items_to_cartitemins(db_items: list["CartItem"]) -> list[CartItemIn]:
    return [
        CartItemIn(product_id=ci.product_id, qty=ci.qty, color=ci.color, model=ci.model)
        for ci in db_items
    ]



#%% APIs
@app.get("/", summary="Health")
def root():
    return {"message": "OK"}

@app.get("/products", response_model=List[ProductByCompatOut],
         summary="List products grouped by (name, compat) with color variants")
def list_products(compat: CompatType | None = Query(default=None, description="Optional phone filter")):
    with SessionLocal() as db:
        q = select(ProductVar)
        if compat:
            q = q.where(ProductVar.compat == compat)
        q = q.order_by(ProductVar.name, ProductVar.compat, ProductVar.colors)
        rows = db.scalars(q).all()

    grouped: dict[tuple[str, str], list[ProductVar]] = {}
    for r in rows:
        grouped.setdefault((r.name, r.compat), []).append(r)

    out: list[ProductByCompatOut] = []
    for (name_key, compat_key), variants in grouped.items():
        base = variants[0]
        price = base.price_cents

        # map color -> (image, db_id, quantity)
        color_to_variant: dict[str, tuple[str, str, int]] = {}
        for v in variants:
            # last write wins per color; if you prefer first write, check with setdefault only if color not in dict
            color_to_variant[v.colors] = (v.image, str(v.id), int(v.quantity or 0))

        total_qty = sum(q for (_, _, q) in color_to_variant.values())

        out.append(ProductByCompatOut(
            id=f"{_slugify(name_key)}--{_slugify(compat_key)}",
            name=name_key,
            compat=compat_key,
            price_cents=price,
            variants=[
                ProductVariantOut(product_id=db_id, colors=c, image=img, quantity=qty)
                for c, (img, db_id, qty) in color_to_variant.items()
            ],
            type=base.type,
            phone=base.phone,
            total_quantity=total_qty,  # NEW (optional, you already had the field)
        ))

    return out

# --- CREATE ONE PRODUCT VARIANT (supports file upload or image_url) ---
@app.post("/product", summary="Create ONE product row (drag&drop upload or URL)")
async def create_single_product_multipart(
    id: str | None = Form(None),
    name: str = Form(...),
    colors: str = Form(...),
    compat: CompatType = Form(...),
    price_cents: conint(ge=0) = Form(...),
    type: str | None = Form(None),
    phone: str | None = Form(None),
    image: UploadFile | None = File(None),
    image_url: str | None = Form(None),
    quantity: conint(ge=0) = Form(0),   # NEW
):
    if not image and not image_url:
        raise HTTPException(status_code=400, detail="Provide either image file or image_url")

    # Upload (or accept URL)
    if image:
        file_bytes = await image.read()
        key = _spaces_key_for(name, image.filename)
        _upload_to_spaces(file_bytes, key, _guess_content_type(image.filename))
        saved_url = _public_url_for(key)
    else:
        saved_url = image_url  # trust full https URL from frontend

    with SessionLocal() as db:
        # If id is not provided, auto-increment the numeric id space as string
        if not id:
            max_id = db.scalar(select(cast(ProductVar.id, Integer)).order_by(cast(ProductVar.id, Integer).desc()))
            id = str((max_id or 0) + 1)

        row = ProductVar(
            id=id,
            name=name,
            image=saved_url,
            colors=colors,
            compat=compat,
            price_cents=int(price_cents),
            type=type,
            phone=phone,
            quantity=int(quantity),   # NEW
        )
        db.add(row)
        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail="Duplicate key violates (id, colors, compat) — row already exists."
            ) from e

    return {
        "status": "created",
        "id": id,
        "name": name,
        "colors": colors,
        "compat": compat,
        "price_cents": int(price_cents),
        "type": type,
        "phone": phone,
        "image": saved_url,
        "quantity": int(quantity),  # NEW
    }

# --- UPDATE ONE PRODUCT VARIANT (by composite key) ---
@app.patch("/product/{id}/{colors}/{compat}", summary="Update ONE product row (by composite key); optionally replace image")
async def update_product_variant(
    id: str = Path(...),
    colors: str = Path(...),
    compat: CompatType = Path(...),

    # Optional fields (multipart) so we can also accept an image upload
    name: Optional[str] = Form(None),
    price_cents: Optional[conint(ge=0)] = Form(None),
    type: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),

    # Image replacement (either upload or URL); if neither given, image is left as-is
    image: UploadFile | None = File(None),
    image_url: str | None = Form(None),

    # NEW: stock update
    quantity: Optional[conint(ge=0)] = Form(None),
):
    with SessionLocal() as db:
        row = db.get(ProductVar, {"id": id, "colors": colors, "compat": compat})
        if not row:
            raise HTTPException(status_code=404, detail="Product variant not found")

        # Replace image if provided
        if image and image.filename:
            file_bytes = await image.read()
            key = _spaces_key_for(name or row.name, image.filename)
            _upload_to_spaces(file_bytes, key, _guess_content_type(image.filename))
            row.image = _public_url_for(key)
        elif image_url:
            row.image = image_url

        # Update simple fields if provided
        if name is not None:
            row.name = name
        if price_cents is not None:
            row.price_cents = int(price_cents)
        if type is not None:
            row.type = type
        if phone is not None:
            row.phone = phone
        if quantity is not None:            # NEW
            row.quantity = int(quantity)

        db.commit()

        return {
            "status": "updated",
            "id": row.id,
            "colors": row.colors,
            "compat": row.compat,
            "name": row.name,
            "price_cents": row.price_cents,
            "type": row.type,
            "phone": row.phone,
            "image": row.image,
            "quantity": row.quantity,       # NEW
        }


@app.delete(
    "/product/{id}/{colors}/{compat}",
    summary="Delete ONE product row (by composite key)"
)
def delete_product_variant(
    id: str = Path(...),
    colors: str = Path(...),
    compat: CompatType = Path(...),
):
    with SessionLocal() as db:
        row = db.get(ProductVar, {"id": id, "colors": colors, "compat": compat})
        if not row:
            raise HTTPException(status_code=404, detail="Product variant not found")

        db.delete(row)
        db.commit()
        return {"status": "deleted", "id": id, "colors": colors, "compat": compat}


@app.post("/checkout/quote", response_model=QuoteOut)
def quote(items: List[CartItemIn]):
    out_items, subtotal, _prod_snap, _prod_index = validate_and_price(items)
    shipping = compute_shipping(subtotal)
    return QuoteOut(items=out_items, subtotal_cents=subtotal, shipping_cents=shipping, total_cents=subtotal+shipping)

@app.post("/orders")
def create_order(payload: CheckoutPayload):
    items = payload.items
    customer = payload.customer

    out_items, subtotal, prod_snap, prod_index = validate_and_price(items)
    if not out_items:
        raise HTTPException(status_code=400, detail="Cart is empty.")

    shipping = compute_shipping(subtotal)
    total = subtotal + shipping

    with SessionLocal() as db:
        order = Order(
            status="PENDING",
            currency="EUR",
            subtotal_cents=subtotal,
            shipping_cents=shipping,
            total_cents=total,
            customer_first_name=customer.first_name,
            customer_last_name=customer.last_name,
            customer_email=customer.email,
            ship_address_line1=customer.address.line1,
            ship_address_line2=customer.address.line2,
            ship_city=customer.address.city,
            ship_postal_code=customer.address.postal_code,
            ship_country=customer.address.country,
        )
        db.add(order)
        db.flush()

        for qi in out_items:
            # choose variant image for this line
            idx = prod_index[qi.product_id]
            image = (
                idx["img_by_variant"].get((qi.color, qi.model)) or
                idx["img_by_color"].get(qi.color) or
                idx["any_image"]
            )

            db.add(OrderItem(
                order_id=order.id,
                product_id=qi.product_id,
                product_name=qi.name,
                image=image,
                color=qi.color,
                model=qi.model,
                qty=qi.qty,
                unit_price_cents=qi.unit_price_cents,
                line_total_cents=qi.line_total_cents,
            ))

        db.commit()

    return {
        "id": order.id,
        "status": order.status,
        "currency": order.currency,
        "subtotal_cents": order.subtotal_cents,
        "shipping_cents": order.shipping_cents,
        "total_cents": order.total_cents,
        "complete": order.complete if order.complete is not None else 0,
        "items": [
            {
                "id": x.id,
                "product_id": x.product_id,
                "product_name": x.product_name,
                "image": x.image,
                "color": x.color,
                "model": x.model,
                "qty": x.qty,
                "unit_price_cents": x.unit_price_cents,
                "line_total_cents": x.line_total_cents,
            }
            for x in order.items
        ],
    }

@app.get(
    "/orders",
    response_model=List[OrderListItemOut],
    summary="List orders (optionally filter by status). Newest first."
)
def list_orders(
    status: str | None = Query(default=None, description="Optional: PENDING|COMPLETED|CANCELED"),
    limit: conint(ge=1, le=500) = 100,
    offset: conint(ge=0) = 0,
):
    with SessionLocal() as db:
        q = select(Order).options(selectinload(Order.items)).order_by(Order.created_at.desc())
        if status:
            q = q.where(Order.status == status.upper())
        q = q.limit(limit).offset(offset)
        rows = db.scalars(q).all()

        out: list[OrderListItemOut] = []
        for order in rows:
            out.append(OrderListItemOut(
                id=order.id,
                status=order.status,
                currency=order.currency,
                subtotal_cents=order.subtotal_cents,
                shipping_cents=order.shipping_cents,
                total_cents=order.total_cents,
                created_at=str(order.created_at) if getattr(order, "created_at", None) else None,
                complete=order.complete,  # ← add
                items=[OrderItemOut.model_validate(x, from_attributes=True) for x in order.items],
            ))
        return out

@app.post("/orders/{order_id}/complete", summary="Mark order as complete (complete=1)")
def mark_order_complete(order_id: str = Path(..., description="Order ID")):
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")

        # idempotent: only write if needed
        if order.complete != 1:
            order.complete = 1
            db.commit()

        # minimal response (doesn't change existing endpoints’ outputs)
        return {"ok": True, "order_id": order_id, "complete": 1}

@app.post("/checkout/session")
def create_checkout_session(payload: CheckoutPayload, request: Request, x_forwarded_for: str | None = Header(None)):
    items = payload.items
    customer = payload.customer

    # 1) Validate & compute pricing (no stock check here)
    out_items, subtotal, _prod_snap, prod_index = validate_and_price(items)
    if not out_items:
        raise HTTPException(status_code=400, detail="Cart is empty.")
    shipping = compute_shipping(subtotal)
    total = subtotal + shipping

    client_ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else (request.client.host if request.client else None))

    # 2) Create Order and place stock reservations atomically
    with SessionLocal() as db:
        # free any expired holds before we compute availability
        release_expired_reservations(db)

        order = Order(
            status="PENDING",
            currency="EUR",
            subtotal_cents=subtotal,
            shipping_cents=shipping,
            total_cents=total,
            customer_first_name=customer.first_name,
            customer_last_name=customer.last_name,
            customer_email=customer.email,
            ship_address_line1=customer.address.line1,
            ship_address_line2=customer.address.line2,
            ship_city=customer.address.city,
            ship_postal_code=customer.address.postal_code,
            ship_country=customer.address.country,
        )
        db.add(order)
        db.flush()
        order_id = order.id

        # Optional: anti-abuse limiter
        try:
            ensure_user_reservation_limits(db, email=customer.email, ip=client_ip, max_active_items=10)
        except HTTPException:
            db.rollback()
            raise

        # For each line: lock row, check qty, decrement, add reservation + order item
        for qi in out_items:
            # lock the variant row
            pv_row = db.execute(
                sa_select(ProductVar)
                .where(
                    ProductVar.id == qi.product_id,
                    ProductVar.colors == qi.color,
                    ProductVar.compat == qi.model
                )
                .with_for_update()
            ).scalar_one_or_none()
            if not pv_row:
                db.rollback()
                raise HTTPException(status_code=400, detail=f"Variant not found for {qi.product_id} / {qi.color} / {qi.model}")

            # ensure stock
            if pv_row.quantity < qi.qty:
                db.rollback()
                raise HTTPException(
                    status_code=409,
                    detail=f"Out of stock for {qi.product_id} ({qi.model} / {qi.color}). Available={pv_row.quantity}, requested={qi.qty}"
                )

            # decrement stock and create reservation
            pv_row.quantity = pv_row.quantity - qi.qty
            db.add(StockReservation(
                order_id=order_id,
                product_id=qi.product_id,
                color=qi.color,
                model=qi.model,
                qty=qi.qty,
                state="HELD",
                expires_at=now_utc() + timedelta(minutes=RES_HOLD_MINUTES),
            ))

            # choose image for this line (as you already do)
            idx = prod_index[qi.product_id]
            image = (
                idx["img_by_variant"].get((qi.color, qi.model))
                or idx["img_by_color"].get(qi.color)
                or idx["any_image"]
            )
            db.add(OrderItem(
                order_id=order_id,
                product_id=qi.product_id,
                product_name=qi.name,
                image=image,
                color=qi.color,
                model=qi.model,
                qty=qi.qty,
                unit_price_cents=qi.unit_price_cents,
                line_total_cents=qi.line_total_cents,
            ))

        db.commit()

    # 3) Create Stripe session (same as you had)
    line_items = []
    for qi in out_items:
        idx = prod_index[qi.product_id]
        image = (
            idx["img_by_variant"].get((qi.color, qi.model))
            or idx["img_by_color"].get(qi.color)
            or idx["any_image"]
        )
        product_data = { "name": f"{qi.name} — {qi.model} — {qi.color}" }
        if isinstance(image, str) and image.startswith(("http://", "https://")):
            product_data["images"] = [image]
        line_items.append({
            "price_data": {
                "currency": "eur",
                "product_data": product_data,
                "unit_amount": qi.unit_price_cents,
            },
            "quantity": qi.qty,
        })
    if shipping > 0:
        line_items.append({
            "price_data": {
                "currency": "eur",
                "product_data": {"name": "Shipping"},
                "unit_amount": shipping,
            },
            "quantity": 1,
        })

    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=f"{FRONTEND_ORIGIN}/success?order_id={order_id}",
        cancel_url=f"{FRONTEND_ORIGIN}/cancel?order_id={order_id}",
        line_items=line_items,
        metadata={"order_id": order_id},
        allow_promotion_codes=True,
        customer_email=customer.email,
        idempotency_key=order_id,
    )

    with SessionLocal() as db:
        db_order = db.get(Order, order_id)
        if db_order:
            db_order.stripe_session_id = session.id
            db.commit()

    return {"checkout_url": session.url, "order_id": order_id}

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    try:
        event = stripe.Webhook.construct_event(payload, sig, endpoint_secret)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    etype = event["type"]
    data = event["data"]["object"]

    if etype == "checkout.session.completed":
        order_id = (data.get("metadata") or {}).get("order_id")
        pi_id = data.get("payment_intent")
        session_id = data.get("id")
        if order_id:
            with SessionLocal() as db:
                order = db.get(Order, order_id)
                if order:
                    order.status = "COMPLETED"
                    order.payment_intent_id = pi_id
                    order.stripe_session_id = session_id
                    # Mark reservations COMMITTED
                    db.execute(
                        update(StockReservation)
                        .where(
                            StockReservation.order_id == order_id,
                            StockReservation.state == "HELD"
                        )
                        .values(state="COMMITTED")
                    )
                    db.commit()

    elif etype in ("checkout.session.expired", "payment_intent.payment_failed"):
        order_id = (data.get("metadata") or {}).get("order_id")
        if order_id:
            with SessionLocal() as db:
                # If the order is still PENDING, mark CANCELED
                order = db.get(Order, order_id)
                if order and order.status == "PENDING":
                    order.status = "CANCELED"
                # Proactively release any HELD reservations for this order (don’t wait for lazy cleaner)
                held = db.scalars(
                    sa_select(StockReservation)
                    .where(StockReservation.order_id==order_id, StockReservation.state=="HELD")
                    .with_for_update(skip_locked=True)
                ).all()
                for r in held:
                    pv = db.get(ProductVar, {"id": r.product_id, "colors": r.color, "compat": r.model})
                    if pv:
                        # lock pv
                        db.execute(
                            sa_select(ProductVar)
                            .where(ProductVar.id==r.product_id, ProductVar.colors==r.color, ProductVar.compat==r.model)
                            .with_for_update()
                        )
                        pv.quantity = pv.quantity + r.qty
                    r.state = "RELEASED"
                db.commit()

    return {"received": True}


# --- Small helper to query order status (optional but handy for success page) ---
@app.get("/orders/{order_id}")
def get_order(order_id: str):
    """
    Fetch an order with items and customer/shipping fields.
    Useful for success page and backoffice.
    """
    with SessionLocal() as db:
        order = db.execute(
            select(Order)
            .options(selectinload(Order.items))
            .where(Order.id == order_id)
        ).scalar_one_or_none()

        if not order:
            raise HTTPException(status_code=404, detail="Order not found")

        return {
            "id": order.id,
            "status": order.status,
            "currency": order.currency,
            "subtotal_cents": order.subtotal_cents,
            "shipping_cents": order.shipping_cents,
            "total_cents": order.total_cents,
            "items": [
                {
                    "id": x.id,
                    "product_id": x.product_id,
                    "product_name": x.product_name,
                    "image": x.image,
                    "color": x.color,
                    "model": x.model,
                    "qty": x.qty,
                    "unit_price_cents": x.unit_price_cents,
                    "line_total_cents": x.line_total_cents,
                }
                for x in order.items
            ],
            "stripe_session_id": order.stripe_session_id,
            "payment_intent_id": order.payment_intent_id,
            "customer": {
                "first_name": order.customer_first_name,
                "last_name": order.customer_last_name,
                "email": order.customer_email,
                "address": {
                    "line1": order.ship_address_line1,
                    "line2": order.ship_address_line2,
                    "city": order.ship_city,
                    "postal_code": order.ship_postal_code,
                    "country": order.ship_country,
                },
            },
            "created_at": str(order.created_at) if getattr(order, "created_at", None) else None,
        }

@app.get("/types", response_model=List[TypeOut], summary="List all product types")
def list_types():
    with SessionLocal() as db:
        rows = db.scalars(select(TypeRow).order_by(TypeRow.name.asc())).all()
        return rows

@app.get("/phones", response_model=List[PhoneOut], summary="List all phones")
def list_phones():
    with SessionLocal() as db:
        rows = db.scalars(select(PhoneRow).order_by(PhoneRow.name.asc())).all()
        return rows

@app.get("/subphones", response_model=List[SubphoneOut], summary="List all subphones (phone + name)")
def list_subphones():
    with SessionLocal() as db:
        rows = db.scalars(select(SubphoneRow).order_by(SubphoneRow.phone.asc(), SubphoneRow.name.asc())).all()
        return rows

@app.post("/types", summary="Create a new type", response_model=TypeOut, status_code=201)
def create_type(body: TypeIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Type name cannot be empty.")
    with SessionLocal() as db:
        row = TypeRow(name=name)
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail="Type already exists.")
        return row

@app.delete("/types/{name}", summary="Delete a type", status_code=204)
def delete_type(name: str = Path(...)):
    key = name.strip()
    with SessionLocal() as db:
        row = db.get(TypeRow, key)
        if not row:
            raise HTTPException(status_code=404, detail="Type not found.")
        db.delete(row)
        db.commit()
    return

@app.post("/phones", summary="Create a new phone", response_model=PhoneOut, status_code=201)
def create_phone(body: PhoneIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Phone name cannot be empty.")
    with SessionLocal() as db:
        row = PhoneRow(name=name)
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail="Phone already exists.")
        return row

@app.delete("/phones/{name}", summary="Delete a phone (cascades subphones)", status_code=204)
def delete_phone(name: str = Path(..., description="Phone name (e.g., iPhone)")):
    key = name.strip()
    with SessionLocal() as db:
        row = db.get(PhoneRow, key)
        if not row:
            raise HTTPException(status_code=404, detail="Phone not found.")
        db.delete(row)  # ON DELETE CASCADE removes related subphones
        db.commit()
    return

@app.post("/subphones", summary="Create a new subphone for a phone", response_model=SubphoneOut, status_code=201)
def create_subphone(body: SubphoneIn):
    phone = body.phone.strip()
    name  = body.name.strip()
    if not phone or not name:
        raise HTTPException(status_code=400, detail="Both phone and name are required.")
    with SessionLocal() as db:
        # Optionally verify phone exists (FK will enforce too)
        if not db.get(PhoneRow, phone):
            raise HTTPException(status_code=400, detail=f"Phone '{phone}' does not exist.")
        row = SubphoneRow(phone=phone, name=name)
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail="Subphone already exists for this phone.")
        return row

@app.delete(
    "/subphones/{phone}/{name}",
    summary="Delete a subphone",
    status_code=204
)
def delete_subphone(
    phone: str = Path(..., description="Parent phone name (e.g., iPhone)"),
    name:  str = Path(..., description="Subphone name (e.g., iPhone 16 Pro)")
):
    p = phone.strip()
    n = name.strip()
    with SessionLocal() as db:
        row = db.get(SubphoneRow, {"phone": p, "name": n})
        if not row:
            raise HTTPException(status_code=404, detail="Subphone not found.")
        db.delete(row)
        db.commit()
    return

@app.get("/debug/spaces/health")
def spaces_health():
    # show which env vars are set (mask secrets)
    cfg = {
        "SPACES_REGION": os.getenv("SPACES_REGION"),
        "SPACES_ENDPOINT": os.getenv("SPACES_ENDPOINT"),
        "SPACES_BUCKET": os.getenv("SPACES_BUCKET"),
        "SPACES_FOLDER": os.getenv("SPACES_FOLDER"),
        "SPACES_CDN_BASE": os.getenv("SPACES_CDN_BASE"),
        "SPACES_KEY_set": bool(os.getenv("SPACES_KEY")),
        "SPACES_SECRET_set": bool(os.getenv("SPACES_SECRET")),
    }
    try:
        # quick round-trip: HEAD bucket
        s3.head_bucket(Bucket=SPACES_BUCKET)
        status = "ok: head_bucket passed"
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"

    return {"status": status, "config": cfg}


@app.get("/cart", response_model=CartOut, summary="Get cart with priced items")
def get_cart(request: Request):
    cid = _require_cart_id(request)
    with SessionLocal() as db:
        db_items = db.execute(
            select(CartItem).where(CartItem.cart_id == cid)
        ).scalars().all()

    items_in = _cart_items_to_cartitemins(db_items)
    if not items_in:
        return CartOut(items=[], subtotal_cents=0, shipping_cents=0, total_cents=0)

    out_items, subtotal, _snap, _idx = validate_and_price(items_in)
    shipping = compute_shipping(subtotal)
    return CartOut(items=out_items, subtotal_cents=subtotal, shipping_cents=shipping, total_cents=subtotal + shipping)


@app.patch("/cart/items", response_model=CartOut, summary="Add/Set/Remove a cart line")
def patch_cart_items(p: CartItemPatch, request: Request):
    cid = _require_cart_id(request)

    with SessionLocal() as db:
        line = db.execute(
            select(CartItem).where(
                CartItem.cart_id == cid,
                CartItem.product_id == p.product_id,
                CartItem.color == p.color,
                CartItem.model == p.model,
            )
        ).scalar_one_or_none()

        if p.action == "add":
            add_qty = max(1, p.qty or 1)
            if line:
                line.qty = min(100, line.qty + add_qty)
            else:
                db.add(CartItem(cart_id=cid, product_id=p.product_id, color=p.color, model=p.model, qty=add_qty))

        elif p.action == "set":
            if p.qty <= 0:
                if line:
                    db.delete(line)
            else:
                if line:
                    line.qty = p.qty
                else:
                    db.add(CartItem(cart_id=cid, product_id=p.product_id, color=p.color, model=p.model, qty=p.qty))

        elif p.action == "remove":
            if line:
                db.delete(line)

        db.commit()

        # Re-read for pricing
        db_items = db.execute(
            select(CartItem).where(CartItem.cart_id == cid)
        ).scalars().all()

    items_in = _cart_items_to_cartitemins(db_items)
    if not items_in:
        return CartOut(items=[], subtotal_cents=0, shipping_cents=0, total_cents=0)

    out_items, subtotal, _snap, _idx = validate_and_price(items_in)
    shipping = compute_shipping(subtotal)
    return CartOut(items=out_items, subtotal_cents=subtotal, shipping_cents=shipping, total_cents=subtotal + shipping)


@app.delete("/cart", summary="Empty cart")
def clear_cart(request: Request):
    cid = _require_cart_id(request)
    with SessionLocal() as db:
        db.query(CartItem).filter(CartItem.cart_id == cid).delete()
        db.commit()
    return {"cleared": True}