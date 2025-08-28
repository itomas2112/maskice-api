
from typing import List
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Header, Request, Query, Path
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, update
import stripe

from app.db.session import get_db
from app.core.config import FRONTEND_ORIGIN, RES_HOLD_MINUTES
from app.models.entities import ProductVar, Order, OrderItem, StockReservation
from app.schemas.types import (
    CartItemIn, CheckoutPayload, QuoteOut, QuoteItemOut, OrderListItemOut, OrderItemOut, OrderOut
)
from app.services.pricing import validate_and_price, compute_shipping
from app.services.reservations import release_expired_reservations, ensure_user_reservation_limits, now_utc

router = APIRouter()

@router.post("/checkout/quote", response_model=QuoteOut)
def quote(payload: List[CartItemIn], db: Session = Depends(get_db)):
    out_items, subtotal, _prod_snap, _prod_index = validate_and_price(db, payload)
    shipping = compute_shipping(subtotal)
    return QuoteOut(items=out_items, subtotal_cents=subtotal, shipping_cents=shipping, total_cents=subtotal+shipping)

@router.post("/orders")
def create_order(payload: CheckoutPayload, db: Session = Depends(get_db)):
    items = payload.items
    customer = payload.customer

    out_items, subtotal, prod_snap, prod_index = validate_and_price(db, items)
    if not out_items:
        raise HTTPException(status_code=400, detail="Cart is empty.")

    shipping = compute_shipping(subtotal)
    total = subtotal + shipping

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
        idx = prod_index[qi.product_id]
        image = idx["img_by_variant"].get((qi.color, qi.model)) or idx["img_by_color"].get(qi.color) or idx["any_image"]
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

@router.get("/orders", response_model=List[OrderListItemOut], summary="List orders (optionally filter by status). Newest first.")
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

@router.post("/orders/{order_id}/complete", summary="Mark order as complete (complete=1)")
def mark_order_complete(order_id: str, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.complete != 1:
        order.complete = 1
        db.commit()
    return {"ok": True, "order_id": order_id, "complete": 1}

@router.post("/checkout/session")
def create_checkout_session(payload: CheckoutPayload, request: Request, db: Session = Depends(get_db), x_forwarded_for: str | None = Header(None)):
    items = payload.items
    customer = payload.customer

    out_items, subtotal, _prod_snap, prod_index = validate_and_price(db, items)
    if not out_items:
        raise HTTPException(status_code=400, detail="Cart is empty.")
    shipping = compute_shipping(subtotal)
    total = subtotal + shipping

    client_ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else (request.client.host if request.client else None))

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

    try:
        ensure_user_reservation_limits(db, email=customer.email, ip=client_ip, max_active_items=10)
    except HTTPException:
        db.rollback()
        raise

    for qi in out_items:
        pv_row = db.execute(
            select(ProductVar).where(
                ProductVar.id == qi.product_id,
                ProductVar.colors == qi.color,
                ProductVar.compat == qi.model
            ).with_for_update()
        ).scalar_one_or_none()
        if not pv_row:
            db.rollback()
            raise HTTPException(status_code=400, detail=f"Variant not found for {qi.product_id} / {qi.color} / {qi.model}")
        if pv_row.quantity < qi.qty:
            db.rollback()
            raise HTTPException(status_code=409, detail=f"Out of stock for {qi.product_id} ({qi.model} / {qi.color}). Available={pv_row.quantity}, requested={qi.qty}")

        pv_row.quantity = pv_row.quantity - qi.qty
        db.add(StockReservation(
            order_id=order_id, product_id=qi.product_id, color=qi.color, model=qi.model,
            qty=qi.qty, state="HELD", expires_at=now_utc() + timedelta(minutes=RES_HOLD_MINUTES),
        ))

        idx = prod_index[qi.product_id]
        image = idx["img_by_variant"].get((qi.color, qi.model)) or idx["img_by_color"].get(qi.color) or idx["any_image"]
        db.add(OrderItem(
            order_id=order_id, product_id=qi.product_id, product_name=qi.name, image=image,
            color=qi.color, model=qi.model, qty=qi.qty, unit_price_cents=qi.unit_price_cents, line_total_cents=qi.line_total_cents,
        ))

    db.commit()

    line_items = []
    for qi in out_items:
        idx = prod_index[qi.product_id]
        image = idx["img_by_variant"].get((qi.color, qi.model)) or idx["img_by_color"].get(qi.color) or idx["any_image"]
        product_data = {"name": f"{qi.name} — {qi.model} — {qi.color}"}
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
            "price_data": {"currency": "eur", "product_data": {"name": "Shipping"}, "unit_amount": shipping},
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

    db_order = db.get(Order, order_id)
    if db_order:
        db_order.stripe_session_id = session.id
        db.commit()

    return {"checkout_url": session.url, "order_id": order_id}

@router.get("/orders/{order_id}")
def get_order(order_id: str, db: Session = Depends(get_db)):
    order = db.execute(
        select(Order).options(selectinload(Order.items)).where(Order.id == order_id)
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
