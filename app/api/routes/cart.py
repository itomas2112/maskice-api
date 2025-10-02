
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import delete as sa_delete

from app.db.session import get_db
from app.api.deps import require_cart_id
from app.models.entities import CartItem, ProductVar
from app.schemas.types import CartItemIn, CartLineOut, CartSummaryOut

router = APIRouter()

@router.get("/cart", response_model=CartSummaryOut, summary="Get cart with totals")
def get_cart(request: Request, db: Session = Depends(get_db)):
    cid = require_cart_id(request)

    rows = (
        db.query(CartItem, ProductVar)
        .join(ProductVar, ProductVar.id == CartItem.product_id)
        .filter(CartItem.cart_id == cid)
        .all()
    )

    items: list[CartLineOut] = []
    subtotal = 0
    for cart_item, product in rows:
        line_total = product.price_cents * cart_item.qty
        subtotal += line_total
        items.append(CartLineOut(
            product_id=cart_item.product_id,
            color=cart_item.color,
            model=cart_item.model,
            qty=cart_item.qty,
            name=product.name,
            unit_price_cents=product.price_cents,
            line_total_cents=line_total,
            image=product.image,
        ))

    # NOTE: Original get_cart uses 20€ threshold (keep exact behavior)
    shipping = 0 if subtotal >= 2000 or subtotal == 0 else 200
    total = subtotal + shipping

    return CartSummaryOut(items=items, subtotal_cents=subtotal, shipping_cents=shipping, total_cents=total)

@router.post("/cart/items", summary="Create or update a cart line")
def upsert_cart_item(p: CartItemIn, request: Request, db: Session = Depends(get_db)):
    cid = require_cart_id(request)

    # 1. Fetch product from products table
    product = db.execute(
        select(ProductVar).where(ProductVar.id == p.product_id)   # 🔧 adjust model/col name
    ).scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # assume your products table has a column called `quantity`
    available_qty = product.quantity

    # 2. Fetch existing cart line
    line = db.execute(
        select(CartItem).where(
            CartItem.cart_id == cid,
            CartItem.product_id == p.product_id,
            CartItem.color == p.color,
            CartItem.model == p.model,
        )
    ).scalar_one_or_none()

    if line:
        # check if new qty would exceed stock
        if line.qty + p.qty > available_qty:
            raise HTTPException(status_code=400, detail="Not enough stock")
        line.qty += p.qty
    else:
        # check if requested qty itself exceeds stock
        if p.qty > available_qty:
            raise HTTPException(status_code=400, detail="Not enough stock")
        line = CartItem(
            cart_id=cid,
            product_id=p.product_id,
            color=p.color,
            model=p.model,
            qty=p.qty,
        )
        db.add(line)

    db.commit()
    db.refresh(line)
    return {"ok": True, "item_id": getattr(line, "id", None)}

@router.delete("/cart/items/{product_id}/{color}/{model}", summary="Remove a product from cart")
def delete_cart_item(request: Request, product_id: str, color: str, model: str, db: Session = Depends(get_db)):
    cid = require_cart_id(request)
    line = db.get(CartItem, (cid, product_id, color, model))
    if line:
        db.delete(line)
        db.commit()
    return {"ok": True}

@router.delete("/cart", summary="Clear entire cart")
def clear_cart(request: Request, db: Session = Depends(get_db)):
    cid = require_cart_id(request)
    # SQLAlchemy 1.4/2.0 compatible bulk delete:
    db.execute(sa_delete(CartItem).where(CartItem.cart_id == cid))
    db.commit()
    return {"ok": True}