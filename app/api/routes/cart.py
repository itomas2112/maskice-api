
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import get_db
from app.api.deps import require_cart_id
from app.models.entities import CartItem, ProductVar
from app.schemas.types import CartItemIn, CartItemPatch, CartLineOut, CartSummaryOut

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

    line = db.execute(
        select(CartItem).where(
            CartItem.cart_id == cid,
            CartItem.product_id == p.product_id,
            CartItem.color == p.color,
            CartItem.model == p.model,
        )
    ).scalar_one_or_none()

    if line:
        line.qty += p.qty
    else:
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

@router.patch("/cart/items", summary="Patch quantity of a cart line")
def patch_cart_item(p: CartItemPatch, request: Request, db: Session = Depends(get_db)):
    cid = require_cart_id(request)

    line = db.execute(
        select(CartItem).where(
            CartItem.cart_id == cid,
            CartItem.product_id == p.product_id,
            CartItem.color == p.color,
            CartItem.model == p.model,
        )
    ).scalar_one_or_none()

    if not line:
        raise HTTPException(status_code=404, detail="Cart item not found")

    line.qty = p.qty
    db.commit()
    db.refresh(line)
    return {"ok": True, "qty": line.qty}

@router.delete("/cart/items/{product_id}/{color}/{model}", summary="Remove a product from cart")
def delete_cart_item(request: Request, product_id: str, color: str, model: str, db: Session = Depends(get_db)):
    cid = require_cart_id(request)
    line = db.get(CartItem, (cid, product_id, color, model))
    if line:
        db.delete(line)
        db.commit()
    return {"ok": True}
