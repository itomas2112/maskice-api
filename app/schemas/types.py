
from typing import List, Literal, Optional
from enum import Enum as PyEnum
from pydantic import BaseModel, conint, EmailStr

CompatType = Literal["iPhone 16", "iPhone 16 Pro"]

class ProductVariantOut(BaseModel):
    product_id: str
    colors: str
    image: str
    quantity: int

class ProductByCompatOut(BaseModel):
    id: str
    name: str
    compat: CompatType
    price_cents: int
    variants: list[ProductVariantOut]
    type: str | None = None
    phone: str | None = None
    total_quantity: int | None = None

class ProductRowIn(BaseModel):
    id: Optional[str] = None
    name: str
    image: str
    colors: str
    compat: CompatType
    price_cents: conint(ge=0)
    type: str
    phone: str

class ProductOut(BaseModel):
    id: str
    name: str
    image: str
    colors: list[str]
    compat: list[CompatType]
    price_cents: int
    class Config: from_attributes = True

class CartItemIn(BaseModel):
    product_id: str
    qty: int
    color: str
    model: CompatType

class QuoteItemOut(BaseModel):
    product_id: str
    name: str
    color: str
    model: CompatType
    qty: int
    unit_price_cents: int
    line_total_cents: int

class QuoteOut(BaseModel):
    items: list[QuoteItemOut]
    subtotal_cents: int
    shipping_cents: int
    total_cents: int

class OrderItemOut(BaseModel):
    id: str
    product_id: str
    product_name: str
    image: str
    color: str
    model: CompatType
    qty: int
    unit_price_cents: int
    line_total_cents: int
    class Config: from_attributes = True

class OrderListItemOut(BaseModel):
    id: str
    status: str
    currency: str
    subtotal_cents: int
    shipping_cents: int
    total_cents: int
    created_at: str | None = None
    complete: int
    items: list[OrderItemOut]
    class Config: from_attributes = True

class OrderOut(BaseModel):
    id: str
    status: str
    currency: str
    subtotal_cents: int
    shipping_cents: int
    total_cents: int
    items: list[OrderItemOut]
    class Config: from_attributes = True

class AddressIn(BaseModel):
    line1: str
    line2: str | None = None
    city: str
    postal_code: str
    country: str

class CustomerIn(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    address: AddressIn

class CheckoutPayload(BaseModel):
    items: list[CartItemIn]
    customer: CustomerIn

class CartItemPatch(BaseModel):
    action: Literal["add", "set", "remove"]
    product_id: str
    color: str
    model: CompatType
    qty: conint(ge=0, le=100) = 0

class CartLineOut(BaseModel):
    product_id: str
    color: str
    model: str
    qty: int
    name: str
    unit_price_cents: int
    line_total_cents: int
    image: str

class CartSummaryOut(BaseModel):
    items: list[CartLineOut]
    subtotal_cents: int
    shipping_cents: int
    total_cents: int

class TypeOut(BaseModel):
    name: str
    class Config: from_attributes = True

class PhoneOut(BaseModel):
    name: str
    class Config: from_attributes = True

class SubphoneOut(BaseModel):
    phone: str
    name: str
    class Config: from_attributes = True

class TypeIn(BaseModel):
    name: str

class PhoneIn(BaseModel):
    name: str

class SubphoneIn(BaseModel):
    phone: str
    name: str

class ReservationState(str, PyEnum):
    HELD = "HELD"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"
