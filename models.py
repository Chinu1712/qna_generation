from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    pdf_downloads = relationship(
        "PDFDownload",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class PDFDownload(Base):
    __tablename__ = "pdf_downloads"

    id = Column(Integer, primary_key=True, index=True)
    pdf_file_name = Column(String, nullable=False)
    pdf_file_path = Column(String, nullable=False)
    question_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="pdf_downloads")