from __future__ import annotations

import uuid
from datetime import datetime
from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class Base(DeclarativeBase):
    pass


class LogChunk(Base):
    __tablename__ = "log_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(String(64), index=True, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))
    created_at = Column(DateTime, default=datetime.utcnow)


@lru_cache(maxsize=1)
def get_engine(database_url: str):
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


def get_session(database_url: str) -> Session:
    return Session(get_engine(database_url))
