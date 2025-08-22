# main.py
import os, uuid
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint
from dotenv import load_dotenv
from pydantic import EmailStr  # top of file

from sqlalchemy import (
    create_engine, Text, Integer, DateTime, select, func as sa_func, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship, selectinload
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import String

import stripe

# ---------- Config ----------
load_dotenv(r'.env')

username = os.getenv("usern")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")

connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}?sslmode=require'

DATABASE_URL = f'postgresql://{username}:{password}@{host}:{port}/{database}?sslmode=require'

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ---------- ORM ----------
class Base(DeclarativeBase):
    pass

class ProductVar(Base):
    __tablename__ = "products"

    # Composite PK so multiple rows with the same product id are allowed
    id: Mapped[str]      = mapped_column(Text, primary_key=True)      # logical product id, e.g. "silicon"
    colors: Mapped[str]  = mapped_column(Text, primary_key=True)      # e.g. "Zelena"
    compat: Mapped[str]  = mapped_column(Text, primary_key=True)      # e.g. "iPhone 16 Pro"

    # Non-key attrs
    name: Mapped[str]        = mapped_column(Text, nullable=False)
    image: Mapped[str]       = mapped_column(Text, nullable=False)    # color/compat-specific image
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(Text, default="PENDING")  # PENDING, PAID, CANCELED
    currency: Mapped[str] = mapped_column(Text, default="EUR")
    subtotal_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    shipping_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    total_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())
    # (optional future fields) customer_email, shipping_address, etc.

    items: Mapped[List["OrderItem"]] = relationship(back_populates="order", cascade="all, delete-orphan")
    stripe_session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    payment_intent_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    customer_first_name: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW
    customer_last_name: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW
    customer_email: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW

    ship_address_line1: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW
    ship_address_line2: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW
    ship_city: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW
    ship_postal_code: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW
    ship_country: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW

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

#%% APIs
@app.get("/", summary="Health")
def root():
    return {"message": "OK"}

@app.get("/products", response_model=List[ProductOut], summary="List all products")
def list_products():
    with SessionLocal() as db:
        rows = db.scalars(select(ProductVar)).all()

    # group by logical product id
    by_id: dict[str, list[ProductVar]] = {}
    for r in rows:
        by_id.setdefault(r.id, []).append(r)

    out: list[ProductOut] = []
    for pid, variants in by_id.items():
        # assume name/price consistent across variants; take from the first
        base = variants[0]
        colors = _uniq([v.colors for v in variants])
        compat = _uniq([v.compat for v in variants])

        # choose a representative image for catalog (first variant)
        rep_image = base.image

        out.append(ProductOut(
            id=pid,
            name=base.name,
            image=rep_image,
            colors=colors,
            compat=compat,  # Pydantic will validate against the Literal
            price_cents=base.price_cents,
        ))
    return out

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

@app.post("/checkout/session")
def create_checkout_session(payload: CheckoutPayload):
    items = payload.items
    customer = payload.customer

    out_items, subtotal, prod_snap, prod_index = validate_and_price(items)
    if not out_items:
        raise HTTPException(status_code=400, detail="Cart is empty.")

    shipping = compute_shipping(subtotal)
    total = subtotal + shipping

    # ... create order as above (use per-variant image when saving OrderItem)

    # 2) Build Stripe line items with the chosen variant image
    line_items = []
    for qi in out_items:
        idx = prod_index[qi.product_id]
        image = (
            idx["img_by_variant"].get((qi.color, qi.model)) or
            idx["img_by_color"].get(qi.color) or
            idx["any_image"]
        )
        line_items.append({
            "price_data": {
                "currency": "eur",
                "product_data": {
                    "name": f"{qi.name} — {qi.model} — {qi.color}",
                    "images": [image],
                },
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

    # 3) Create Stripe Checkout Session
    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=f"{FRONTEND_ORIGIN}/success?order_id={order_id}",
        cancel_url=f"{FRONTEND_ORIGIN}/cancel?order_id={order_id}",
        line_items=line_items,
        metadata={"order_id": order_id},       # keep order_id in metadata
        idempotency_key=order_id,              # makes retries safe
        allow_promotion_codes=True,
        customer_email=customer.email,         # prefill email
        # If you prefer Stripe to collect the shipping address too, you can enable:
        # shipping_address_collection={"allowed_countries": [customer.address.country]},
    )

    # 4) Store Stripe IDs on the order
    with SessionLocal() as db:
        db_order = db.get(Order, order_id)
        if db_order:
            db_order.stripe_session_id = session.id
            db_order.payment_intent_id = session.get("payment_intent")
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
                    db.commit()

    elif etype in ("checkout.session.expired",):
        order_id = (data.get("metadata") or {}).get("order_id")
        if order_id:
            with SessionLocal() as db:
                order = db.get(Order, order_id)
                if order and order.status == "PENDING":
                    order.status = "CANCELED"
                    db.commit()

    # (Optional) handle payment_intent.payment_failed if you use PaymentIntents directly

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
