
from typing import List, Tuple, Dict

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas.types import CartItemIn, QuoteItemOut
from app.models.entities import ProductVar

def compute_shipping(subtotal_cents: int) -> int:
    # Same rule as original helper: free if subtotal >= 25€ or subtotal == 0, else 2€
    return 0 if (subtotal_cents == 0 or subtotal_cents >= 2500) else 200

def validate_and_price(db: Session, items: List[CartItemIn]):
    if not items:
        return [], 0, {}, {}

    wanted_ids = list({i.product_id for i in items})
    variants = db.scalars(select(ProductVar).where(ProductVar.id.in_(wanted_ids))).all()

    prod_index: dict[str, dict] = {}
    for v in variants:
        entry = prod_index.setdefault(v.id, {
            "name": v.name,
            "price_cents": v.price_cents,
            "colors": set(),
            "compat": set(),
            "img_by_variant": {},
            "img_by_color": {},
            "any_image": v.image,
        })
        entry["colors"].add(v.colors)
        entry["compat"].add(v.compat)
        entry["img_by_variant"][(v.colors, v.compat)] = v.image
        entry["img_by_color"].setdefault(v.colors, v.image)

    out_items: list[QuoteItemOut] = []
    subtotal = 0
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

        image = (
            entry["img_by_variant"].get((it.color, it.model)) or
            entry["img_by_color"].get(it.color) or
            entry["any_image"]
        )

        prod_snap[it.product_id] = {
            "id": it.product_id,
            "name": entry["name"],
            "image": image,
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

    return out_items, subtotal, prod_snap, prod_index
