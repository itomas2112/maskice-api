
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Form, UploadFile, File, Path, Query, Depends
from pydantic import conint
from sqlalchemy.orm import Session
from sqlalchemy import select, cast, Integer
from sqlalchemy.exc import IntegrityError

from app.db.session import get_db
from app.models.entities import ProductVar
from app.schemas.types import (
    CompatType, ProductByCompatOut, ProductVariantOut
)
from app.utils.slugify import slugify
from app.services.spaces import upload_to_spaces, spaces_key_for, public_url_for, guess_content_type
from app.api.deps import require_admin

router = APIRouter()

@router.get("/products", response_model=List[ProductByCompatOut], summary="List products grouped by (name, compat) with color variants")
def list_products(compat: CompatType | None = Query(default=None), db: Session = Depends(get_db)):
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

        color_to_variant: dict[str, tuple[str, str, int]] = {}
        for v in variants:
            color_to_variant[v.colors] = (v.image, str(v.id), int(v.quantity or 0))

        total_qty = sum(q for (_, _, q) in color_to_variant.values())

        out.append(ProductByCompatOut(
            id=f"{slugify(name_key)}--{slugify(compat_key)}",
            name=name_key,
            compat=compat_key,  # type: ignore
            price_cents=price,
            variants=[
                ProductVariantOut(product_id=db_id, colors=c, image=img, quantity=qty)
                for c, (img, db_id, qty) in color_to_variant.items()
            ],
            type=base.type,
            phone=base.phone,
            total_quantity=total_qty,
        ))

    return out

@router.post("/product", dependencies=[Depends(require_admin)])
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
    quantity: conint(ge=0) = Form(0),
    db: Session = Depends(get_db),
):
    if not image and not image_url:
        raise HTTPException(status_code=400, detail="Provide either image file or image_url")

    if image:
        file_bytes = await image.read()
        key = spaces_key_for(name, image.filename)
        upload_to_spaces(file_bytes, key, guess_content_type(image.filename))
        saved_url = public_url_for(key)
    else:
        saved_url = image_url  # trust full https URL

    if not id:
        max_id = db.scalar(select(cast(ProductVar.id, Integer)).order_by(cast(ProductVar.id, Integer).desc()))
        id = str((max_id or 0) + 1)

    row = ProductVar(
        id=id, name=name, image=saved_url, colors=colors, compat=compat,  # type: ignore
        price_cents=int(price_cents), type=type, phone=phone, quantity=int(quantity),
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
        "quantity": int(quantity),
    }

@router.patch("/product/{id}/{colors}/{compat}", dependencies=[Depends(require_admin)])
async def update_product_variant(
    id: str = Path(...),
    colors: str = Path(...),
    compat: CompatType = Path(...),
    name: str | None = Form(None),
    price_cents: conint(ge=0) | None = Form(None),
    type: str | None = Form(None),
    phone: str | None = Form(None),
    image: UploadFile | None = File(None),
    image_url: str | None = Form(None),
    quantity: conint(ge=0) | None = Form(None),
    db: Session = Depends(get_db),
):
    row = db.get(ProductVar, {"id": id, "colors": colors, "compat": compat})
    if not row:
        raise HTTPException(status_code=404, detail="Product variant not found")

    if image and image.filename:
        file_bytes = await image.read()
        key = spaces_key_for(name or row.name, image.filename)
        upload_to_spaces(file_bytes, key, guess_content_type(image.filename))
        row.image = public_url_for(key)
    elif image_url:
        row.image = image_url

    if name is not None:
        row.name = name
    if price_cents is not None:
        row.price_cents = int(price_cents)
    if type is not None:
        row.type = type
    if phone is not None:
        row.phone = phone
    if quantity is not None:
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
        "quantity": row.quantity,
    }

@router.delete("/product/{id}/{colors}/{compat}", dependencies=[Depends(require_admin)])
def delete_product_variant(id: str, colors: str, compat: CompatType, db: Session = Depends(get_db)):
    row = db.get(ProductVar, {"id": id, "colors": colors, "compat": compat})
    if not row:
        raise HTTPException(status_code=404, detail="Product variant not found")
    db.delete(row)
    db.commit()
    return {"status": "deleted", "id": id, "colors": colors, "compat": compat}
