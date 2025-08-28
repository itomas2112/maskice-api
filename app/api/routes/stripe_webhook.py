
from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select as sa_select, update

import stripe

from app.core.config import STRIPE_WEBHOOK_SECRET
from app.db.session import get_db
from app.models.entities import Order, StockReservation, ProductVar

router = APIRouter()

@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    endpoint_secret = STRIPE_WEBHOOK_SECRET

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
            order = db.get(Order, order_id)
            if order:
                order.status = "COMPLETED"
                order.payment_intent_id = pi_id
                order.stripe_session_id = session_id
                db.execute(
                    update(StockReservation)
                    .where(StockReservation.order_id == order_id, StockReservation.state == "HELD")
                    .values(state="COMMITTED")
                )
                db.commit()

    elif etype in ("checkout.session.expired", "payment_intent.payment_failed"):
        order_id = (data.get("metadata") or {}).get("order_id")
        if order_id:
            order = db.get(Order, order_id)
            if order and order.status == "PENDING":
                order.status = "CANCELED"
            held = db.scalars(
                sa_select(StockReservation)
                .where(StockReservation.order_id==order_id, StockReservation.state=="HELD")
                .with_for_update(skip_locked=True)
            ).all()
            for r in held:
                pv = db.get(ProductVar, {"id": r.product_id, "colors": r.color, "compat": r.model})
                if pv:
                    db.execute(
                        sa_select(ProductVar)
                        .where(ProductVar.id==r.product_id, ProductVar.colors==r.color, ProductVar.compat==r.model)
                        .with_for_update()
                    )
                    pv.quantity = pv.quantity + r.qty
                r.state = "RELEASED"
            db.commit()

    return {"received": True}
