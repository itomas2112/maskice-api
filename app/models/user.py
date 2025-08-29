from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Text, DateTime
from sqlalchemy.sql import func as sa_func

from app.models.base import Base

class User(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(Text, primary_key=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="ADMIN", server_default="ADMIN")

