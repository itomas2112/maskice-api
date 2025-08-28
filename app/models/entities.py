
import uuid
from datetime import datetime
from sqlalchemy import Text, Integer, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func as sa_func

from app.models.base import Base

class ProductVar(Base):
    __tablename__ = "products"
    id: Mapped[str]      = mapped_column(Text, primary_key=True)
    colors: Mapped[str]  = mapped_column(Text, primary_key=True)
    compat: Mapped[str]  = mapped_column(Text, primary_key=True)

    name: Mapped[str]        = mapped_column(Text, nullable=False)
    image: Mapped[str]       = mapped_column(Text, nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str | None]  = mapped_column(Text, nullable=True)
    phone: Mapped[str | None] = mapped_column(Text, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(Text, default="PENDING")
    currency: Mapped[str] = mapped_column(Text, default="EUR")
    subtotal_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    shipping_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    total_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())

    complete: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    stripe_session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    payment_intent_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    customer_first_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    customer_last_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    customer_email: Mapped[str | None] = mapped_column(Text, nullable=True)

    ship_address_line1: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_address_line2: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_city: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_postal_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    ship_country: Mapped[str | None] = mapped_column(Text, nullable=True)

class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(Text, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)

    product_id: Mapped[str] = mapped_column(Text, nullable=False)
    product_name: Mapped[str] = mapped_column(Text, nullable=False)
    image: Mapped[str] = mapped_column(Text, nullable=False)

    color: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)

    unit_price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    line_total_cents: Mapped[int] = mapped_column(Integer, nullable=False)

    order = relationship("Order", back_populates="items")

class CartItem(Base):
    __tablename__ = "cart_items"
    cart_id:    Mapped[str] = mapped_column(Text, primary_key=True)
    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    color:      Mapped[str] = mapped_column(Text, primary_key=True)
    model:      Mapped[str] = mapped_column(Text, primary_key=True)
    qty:        Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now(), onupdate=sa_func.now())

class TypeRow(Base):
    __tablename__ = "types"
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())

class PhoneRow(Base):
    __tablename__ = "phones"
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())

class SubphoneRow(Base):
    __tablename__ = "subphones"
    phone: Mapped[str] = mapped_column(Text, ForeignKey("phones.name", onupdate="CASCADE", ondelete="CASCADE"), primary_key=True)
    name:  Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())

class StockReservation(Base):
    __tablename__ = "stock_reservations"
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(Text, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)

    product_id: Mapped[str] = mapped_column(Text, nullable=False)
    color: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)

    state: Mapped[str] = mapped_column(Text, nullable=False, default="HELD", server_default="HELD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

# Helpful index (as in original code)
Index(
    "ix_reservations_active",
    StockReservation.product_id, StockReservation.color, StockReservation.model,
    StockReservation.state, StockReservation.expires_at
)
