
from datetime import datetime, timezone, timedelta
from sqlalchemy import select as sa_select, update, and_, or_, func as sa_func
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.entities import StockReservation, ProductVar, Order

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def release_expired_reservations(db: Session):
    expired = db.scalars(
        sa_select(StockReservation)
        .where(StockReservation.state == "HELD", StockReservation.expires_at <= now_utc())
        .with_for_update(skip_locked=True)
    ).all()
    if not expired:
        return

    by_key: dict[tuple[str,str,str], int] = {}
    for r in expired:
        by_key[(r.product_id, r.color, r.model)] = by_key.get((r.product_id, r.color, r.model), 0) + r.qty

    for (pid, color, model), qty in by_key.items():
        pv = db.get(ProductVar, {"id": pid, "colors": color, "compat": model})
        if pv:
            db.execute(
                sa_select(ProductVar)
                .where(ProductVar.id == pid, ProductVar.colors == color, ProductVar.compat == model)
                .with_for_update()
            )
            pv.quantity = pv.quantity + qty

    for r in expired:
        r.state = "RELEASED"

def ensure_user_reservation_limits(db: Session, email: str | None, ip: str | None, max_active_items: int = 10):
    q = sa_select(sa_func.coalesce(sa_func.sum(StockReservation.qty), 0)).join(
        Order, Order.id == StockReservation.order_id
    ).where(
        StockReservation.state == "HELD",
        Order.status == "PENDING",
        or_(
            and_(email is not None, Order.customer_email == email),
            and_(ip is not None, Order.ship_address_line1 == None, True),
        )
    )
    active_qty = db.scalar(q) or 0
    if active_qty > max_active_items:
        raise HTTPException(status_code=429, detail="Too many active holds. Complete or wait for holds to expire.")
