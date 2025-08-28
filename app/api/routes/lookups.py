
from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.session import get_db
from app.models.entities import TypeRow, PhoneRow, SubphoneRow
from app.schemas.types import TypeOut, PhoneOut, SubphoneOut, TypeIn, PhoneIn, SubphoneIn

router = APIRouter()

@router.get("/types", response_model=list[TypeOut], summary="List all product types")
def list_types(db: Session = Depends(get_db)):
    rows = db.scalars(select(TypeRow).order_by(TypeRow.name.asc())).all()
    return rows

@router.get("/phones", response_model=list[PhoneOut], summary="List all phones")
def list_phones(db: Session = Depends(get_db)):
    rows = db.scalars(select(PhoneRow).order_by(PhoneRow.name.asc())).all()
    return rows

@router.get("/subphones", response_model=list[SubphoneOut], summary="List all subphones (phone + name)")
def list_subphones(db: Session = Depends(get_db)):
    rows = db.scalars(select(SubphoneRow).order_by(SubphoneRow.phone.asc(), SubphoneRow.name.asc())).all()
    return rows

@router.post("/types", summary="Create a new type", response_model=TypeOut, status_code=201)
def create_type(body: TypeIn, db: Session = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Type name cannot be empty.")
    row = TypeRow(name=name)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Type already exists.")
    return row

@router.delete("/types/{name}", summary="Delete a type", status_code=204)
def delete_type(name: str = Path(...), db: Session = Depends(get_db)):
    key = name.strip()
    row = db.get(TypeRow, key)
    if not row:
        raise HTTPException(status_code=404, detail="Type not found.")
    db.delete(row)
    db.commit()
    return

@router.post("/phones", summary="Create a new phone", response_model=PhoneOut, status_code=201)
def create_phone(body: PhoneIn, db: Session = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Phone name cannot be empty.")
    row = PhoneRow(name=name)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Phone already exists.")
    return row

@router.delete("/phones/{name}", summary="Delete a phone (cascades subphones)", status_code=204)
def delete_phone(name: str = Path(..., description="Phone name (e.g., iPhone)"), db: Session = Depends(get_db)):
    key = name.strip()
    row = db.get(PhoneRow, key)
    if not row:
        raise HTTPException(status_code=404, detail="Phone not found.")
    db.delete(row)
    db.commit()
    return

@router.post("/subphones", summary="Create a new subphone for a phone", response_model=SubphoneOut, status_code=201)
def create_subphone(body: SubphoneIn, db: Session = Depends(get_db)):
    phone = body.phone.strip()
    name  = body.name.strip()
    if not phone or not name:
        raise HTTPException(status_code=400, detail="Both phone and name are required.")
    if not db.get(PhoneRow, phone):
        raise HTTPException(status_code=400, detail=f"Phone '{phone}' does not exist.")
    row = SubphoneRow(phone=phone, name=name)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Subphone already exists for this phone.")
    return row

@router.delete("/subphones/{phone}/{name}", summary="Delete a subphone", status_code=204)
def delete_subphone(phone: str = Path(..., description="Parent phone name (e.g., iPhone)"),
                    name: str = Path(..., description="Subphone name (e.g., iPhone 16 Pro)"),
                    db: Session = Depends(get_db)):
    p = phone.strip()
    n = name.strip()
    row = db.get(SubphoneRow, {"phone": p, "name": n})
    if not row:
        raise HTTPException(status_code=404, detail="Subphone not found.")
    db.delete(row)
    db.commit()
    return
