"""
Docstring for api.tables

Module for table declaration for PostgreSQL database using SQLAlchemy
"""
from sqlalchemy import Column, Integer, String, DDL, event, Index
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import VECTOR

Base = declarative_base()

def create_ticket_model(embedding_dim: int):
    class Ticket(Base):
        __tablename__ = "tickets"

        id = Column(Integer, primary_key=True, index=True)
        short_description = Column(String, nullable=False)
        category = Column(String, default="N/D")
        subcategory = Column(String, default="N/D")
        content = Column(String, default="N/D")
        software = Column(String, default="N/D")
        embedding = Column(VECTOR(embedding_dim), nullable=False)

        __table_args__ = (
            # 2. Automatically create an IVFFlat index on embedding
            Index(
                'tickets_embedding_idx',
                'embedding',
                postgresql_using='ivfflat',
                postgresql_with={'lists': 100},
                postgresql_ops={'embedding': 'vector_ip_ops'},
            ),
        )

    return Ticket