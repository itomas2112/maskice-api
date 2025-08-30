# reservations.py
from datetime import datetime, timezone, timedelta
from sqlalchemy import select as sa_select, update, and_, or_, func as sa_func
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.entities import StockReservation, ProductVar, Order

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def release_expired_reservations(db: Session):
    """Find expired HELD reservations, restore stock, mark RELEASED."""
    expired = db.scalars(
        sa_select(StockReservation)
        .where(
            StockReservation.state == "HELD",
            StockReservation.expires_at <= now_utc(),
        )
        .with_for_update(skip_locked=True)
    ).all()
    if not expired:
        return

    # restore grouped to reduce row locks
    for r in expired:
        pv = db.get(ProductVar, {"id": r.product_id, "colors": r.color, "compat": r.model})
        if pv:
            # row lock to avoid races
            db.execute(
                sa_select(ProductVar)
                .where(ProductVar.id == r.product_id,
                       ProductVar.colors == r.color,
                       ProductVar.compat == r.model)
                .with_for_update(skip_locked=True)
            )
            pv.quantity = (pv.quantity or 0) + r.qty
        r.state = "RELEASED"

def ensure_user_reservation_limits(
    db: Session,
    email: str | None,
    ip: str | None,
    max_active_items: int = 10,
):
    """
    Rate limit by OR(email, ip) across PENDING orders with HELD reservations.
    Requires Order.client_ip (nullable str). If you haven't added it yet,
    see migration stub below.
    """
    conds = []
    if email:
        conds.append(Order.customer_email == email)
    if ip:
        # requires new nullable column on Order
        conds.append(Order.client_ip == ip)

    if not conds:
        return  # nothing to check

    q = sa_select(sa_func.coalesce(sa_func.sum(StockReservation.qty), 0)).join(
        Order, Order.id == StockReservation.order_id
    ).where(
        StockReservation.state == "HELD",
        Order.status == "PENDING",
        or_(*conds),
    )

    active_qty = db.scalar(q) or 0
    if active_qty > max_active_items:
        raise HTTPException(
            status_code=429,
            detail="Too many active holds. Complete a payment or wait for holds to expire."
        )
