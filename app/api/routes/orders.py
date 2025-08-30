
from typing import List, Optional
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Header, Request, Query, Path
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, update
from sqlalchemy import select as sa_select
import stripe
import hmac, hashlib, base64, time
import os

from app.db.session import get_db
from app.core.config import FRONTEND_ORIGIN, RES_HOLD_MINUTES
from app.models.entities import ProductVar, Order, OrderItem, StockReservation
from app.schemas.types import (
    CartItemIn, CheckoutPayload, QuoteOut, QuoteItemOut, OrderListItemOut, OrderItemOut, OrderOut
)
from app.services.pricing import validate_and_price, compute_shipping
from app.services.reservations import release_expired_reservations, ensure_user_reservation_limits, now_utc
from app.api.deps import require_admin   # 👈 add this

router = APIRouter()

CANCEL_TOKEN_SECRET = os.environ.get("CANCEL_TOKEN_SECRET", "dev-please-change-me").encode()

def _get_client_ip(request: Request) -> str | None:
    xfwd = request.headers.get("x-forwarded-for")
    if xfwd:
        # first IP in the list
        return xfwd.split(",")[0].strip()
    return request.client.host if request.client else None

def _sign_cancel_token(order_id: str, exp_ts: int) -> str:
    msg = f"{order_id}.{exp_ts}".encode()
    mac = hmac.new(CANCEL_TOKEN_SECRET, msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")

def _verify_cancel_token(order_id: str, exp_ts: int, token: str) -> bool:
    # constant-time compare
    try:
        expected = _sign_cancel_token(order_id, exp_ts)
        # re-add stripped padding for compare safety
        return hmac.compare_digest(expected, token)
    except Exception:
        return False

@router.get("/orders", dependencies=[Depends(require_admin)], response_model=List[OrderListItemOut], summary="List orders (optionally filter by status). Newest first.")
def list_orders(status: str | None = Query(default=None), limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
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
            complete=order.complete,
            items=[OrderItemOut.model_validate(x, from_attributes=True) for x in order.items],
        ))
    return out

@router.post("/orders/{order_id}/complete", dependencies=[Depends(require_admin)], summary="Mark order as complete (complete=1)")
def mark_order_complete(order_id: str, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.complete != 1:
        order.complete = 1
        final_order_statues = 1
        db.commit()
    elif order.complete == 1:
        order.complete = 0
        final_order_statues = 0
        db.commit()
    return {"ok": True, "order_id": order_id, "complete": final_order_statues}

@router.post("/checkout/session")
def create_checkout_session(payload: CheckoutPayload, request: Request, db: Session = Depends(get_db)):
    # 1) Reap expired holds first
    release_expired_reservations(db)

    # 2) Price & validate cart
    items, subtotal, prod_snap, prod_index = validate_and_price(db, payload.items)
    shipping_cents = compute_shipping(subtotal)

    # 3) Abuse limiter BEFORE reserving anything
    client_ip = _get_client_ip(request)
    ensure_user_reservation_limits(db, email=payload.customer.email, ip=client_ip, max_active_items=10)

    # 4) Extract nested address safely
    addr = payload.customer.address
    line1 = getattr(addr, "line1", None) or getattr(addr, "address_line1", None)
    line2 = getattr(addr, "line2", None) or getattr(addr, "address_line2", None)
    city = getattr(addr, "city", None)
    postal_code = getattr(addr, "postal_code", None) or getattr(addr, "zip", None)
    country = getattr(addr, "country", None)

    if not (line1 and city and postal_code and country):
        raise HTTPException(status_code=400, detail="Incomplete shipping address")

    # 5) Create an Order (PENDING) with your exact column names
    order = Order(
        status="PENDING",
        customer_email=payload.customer.email,
        customer_first_name=payload.customer.first_name,
        customer_last_name=payload.customer.last_name,
        ship_address_line1=line1,
        ship_address_line2=line2,
        ship_city=city,
        ship_postal_code=postal_code,
        ship_country=country,
        subtotal_cents=subtotal,
        shipping_cents=shipping_cents,
        total_cents=subtotal + shipping_cents,
        client_ip=client_ip,
    )
    db.add(order)
    db.flush()
    order_id = order.id

    # 6) Reserve stock immediately (row-lock, decrement, HELD)
    expires_at = now_utc() + timedelta(minutes=RES_HOLD_MINUTES)
    for it in items:
        pv = db.execute(
            select(ProductVar)
            .where(
                ProductVar.id == it.product_id,
                ProductVar.colors == it.color,
                ProductVar.compat == it.model,
            )
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if not pv:
            db.rollback()
            raise HTTPException(
                status_code=400,
                detail=f"Variant not found for {it.product_id} / {it.color} / {it.model}",
            )
        if (pv.quantity or 0) < it.qty:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail=f"Out of stock for {it.product_id} ({it.color}/{it.model}). "
                       f"Available={pv.quantity}, requested={it.qty}",
            )

        pv.quantity = pv.quantity - it.qty

        db.add(StockReservation(
            order_id=order_id,
            product_id=it.product_id,
            color=it.color,
            model=it.model,
            qty=it.qty,
            state="HELD",
            expires_at=expires_at,
        ))

        # Snapshot into OrderItem (use your exact column names)
        entry = prod_index[it.product_id]
        image = (
            entry["img_by_variant"].get((it.color, it.model))
            or entry["img_by_color"].get(it.color)
            or entry["any_image"]
        )

        db.add(OrderItem(
            order_id=order_id,
            product_id=it.product_id,
            product_name=entry["name"],
            image=image,
            color=it.color,
            model=it.model,
            qty=it.qty,
            unit_price_cents=it.unit_price_cents,
            line_total_cents=it.line_total_cents,
        ))

    db.commit()  # order + reservations committed

    # 7) Create Stripe Checkout Session (expires >= 30min per Stripe rules)
    try:
        line_items = []
        for it in items:
            entry = prod_index[it.product_id]
            img = (
                entry["img_by_variant"].get((it.color, it.model))
                or entry["img_by_color"].get(it.color)
                or entry["any_image"]
            )

            line_items.append({
                "quantity": it.qty,
                "price_data": {
                    "currency": "eur",
                    "unit_amount": it.unit_price_cents,
                    "product_data": {
                        "name": entry["name"],
                        "images": [img] if isinstance(img, str) and img.startswith("http") else [],
                        "metadata": {"product_id": it.product_id, "color": it.color, "model": it.model},
                    },
                },
            })

        if shipping_cents > 0:
            line_items.append({
                "quantity": 1,
                "price_data": {
                    "currency": "eur",
                    "unit_amount": shipping_cents,
                    "product_data": {"name": "Shipping"},
                },
            })

        # Stripe requires >= 30 minutes
        expires_seconds = max(RES_HOLD_MINUTES, 30) * 60
        exp_ts = int(time.time()) + expires_seconds
        cancel_token = _sign_cancel_token(order_id, exp_ts)

        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=line_items,
            success_url=f"{FRONTEND_ORIGIN}/success?order_id={order_id}",
            cancel_url=f"{FRONTEND_ORIGIN}/cancel?order_id={order_id}&exp={exp_ts}&ct={cancel_token}",
            allow_promotion_codes=True,
            metadata={"order_id": order_id},
            payment_intent_data={"metadata": {"order_id": order_id}},
            expires_at=int(time.time()) + expires_seconds,
            customer_email=payload.customer.email
        )

        # Save session id on the order
        od = db.get(Order, order_id)
        od.stripe_session_id = session.id
        db.commit()

        return {"checkout_url": session.url, "order_id": order_id}

    except Exception as e:
        # Stripe failed — restore stock and cancel the order
        od = db.get(Order, order_id)
        if od and od.status == "PENDING":
            held = db.scalars(
                select(StockReservation)
                .where(
                    StockReservation.order_id == order_id,
                    StockReservation.state == "HELD",
                )
                .with_for_update(skip_locked=True)
            ).all()
            for r in held:
                pv = db.get(ProductVar, {"id": r.product_id, "colors": r.color, "compat": r.model})
                if pv:
                    db.execute(
                        select(ProductVar)
                        .where(
                            ProductVar.id == r.product_id,
                            ProductVar.colors == r.color,
                            ProductVar.compat == r.model,
                        )
                        .with_for_update(skip_locked=True)
                    )
                    pv.quantity = (pv.quantity or 0) + r.qty
                r.state = "RELEASED"

            od.status = "CANCELED"
            db.commit()

        raise HTTPException(status_code=502, detail="Could not create Stripe session") from e

@router.get("/orders/{order_id}") ## Need to be part of the checkout
def get_order(order_id: str, db: Session = Depends(get_db)):
    order = db.execute(
        select(Order).options(selectinload(Order.items)).where(Order.id == order_id)
    ).scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    return {"status": order.status}

@router.post("/orders/{order_id}/cancel")
def cancel_order(
    order_id: str,
    exp: int = Query(..., description="expiry timestamp"),
    ct: str = Query(..., description="cancel token"),
    db: Session = Depends(get_db),
):
    # 1) Verify token & expiry
    now = int(time.time())
    if exp < now or not _verify_cancel_token(order_id, exp, ct):
        raise HTTPException(status_code=403, detail="Invalid or expired cancel token")

    # 2) Standard idempotent cancel
    od = db.get(Order, order_id)
    if not od:
        raise HTTPException(status_code=404, detail="Order not found")
    if od.status != "PENDING":
        return {"order_id": order_id, "status": od.status}

    held = db.scalars(
        sa_select(StockReservation)
        .where(StockReservation.order_id == order_id,
               StockReservation.state == "HELD")
        .with_for_update(skip_locked=True)
    ).all()

    for r in held:
        pv = db.get(ProductVar, {"id": r.product_id, "colors": r.color, "compat": r.model})
        if pv:
            db.execute(
                sa_select(ProductVar)
                .where(ProductVar.id == r.product_id,
                       ProductVar.colors == r.color,
                       ProductVar.compat == r.model)
                .with_for_update(skip_locked=True)
            )
            pv.quantity = (pv.quantity or 0) + r.qty
        r.state = "RELEASED"

    od.status = "CANCELED"
    db.commit()
    return {"order_id": order_id, "status": "CANCELED"}
