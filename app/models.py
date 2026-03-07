"""
SQLAlchemy ORM models for the report_assistant pipeline.

These models represent the database schema for documents and risk factors.
Separate from Pydantic models in data_classes.py (which handle config/validation).
"""

from __future__ import annotations

from pathlib import Path
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Index

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("company", "fiscal_year", name="uq_company_year"),
    )
    id = Column(String, primary_key=True)
    company = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=True)
    fiscal_year = Column(Integer, nullable=False, index=True)
    source_file_path = Column(String, nullable=False)
    questions_file_path = Column(String, nullable=True)
    text_dir = Column(String, nullable=True)
    chunks_dir = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationship to risk factors
    risk_factors = relationship("RiskFactor", back_populates="document", cascade="all, delete-orphan")


class RiskFactor(Base):
    """
    Risk factor chunks extracted from a document.
    
    Each factor represents a single risk factor section with its associated text.
    The idx field tracks the sequential position of the factor within the document.
    """
    __tablename__ = "risk_factors"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    risk_factor = Column(Text, nullable=True)
    text = Column(Text, nullable=False)
    idx = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationship to parent document
    document = relationship("Document", back_populates="risk_factors")

