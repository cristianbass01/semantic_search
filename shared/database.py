"""
Docstring for api.database

Module to manage the connection to the database.
"""
from sqlalchemy import create_engine, text, DDL, event
from sqlalchemy.orm import sessionmaker, Session
from typing import List
import logging
from contextlib import contextmanager

# Custom
from embedding import get_embedding_dim, embedding_is_ready
from tables import Base, create_ticket_model

import config

engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Ticket = None

log = logging.getLogger("uvicorn")

def db_start():
    global Ticket

    log.info("Starting database creation.")
    embedding_dim = get_embedding_dim()

    Ticket = create_ticket_model(embedding_dim)

    # 1. Enable pgvector extension
    log.info("Enabling extension.")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # Create All Tables
    log.info("Creating Ticket table.")
    Base.metadata.create_all(bind=engine)
    

# Function to get a connection to the db and close it after
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_add_ticket(db: Session, ticket_dict: dict, embedding: List[float]):
    # Create a ticket object
    ticket = Ticket(**ticket_dict, embedding=embedding)
    
    # Add and commit to the DB
    db.add(ticket)
    db.commit()
    return

def db_add_tickets_batch(db: Session, tickets: List[dict], embeddings: List[List[float]]):
    if len(tickets) != len(embeddings):
        raise ValueError("tickets and embeddings length mismatch")

    ticket_objs = [
        Ticket(**ticket_dict, embedding=embedding)
        for ticket_dict, embedding in zip(tickets, embeddings)
    ]

    db.add_all(ticket_objs)
    db.commit()
    return

def db_delete_ticket(db: Session, ticket_id: int):
    # Get the ticket object
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        return False
    
    # Delete and commit
    db.delete(ticket)
    db.commit()
    return True

def db_update_ticket(db: Session, ticket_id: int, ticket_dict: dict, embedding: List[float]):
    # Get the ticket object
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        return
    
    # Update the ticket with the new data
    for key, value in ticket_dict.items():
        setattr(ticket, key, value)
    if embedding is not None:
        ticket.embedding = embedding
    
    # Commit to the DB
    db.commit()
    return 

def db_search_tickets(db: Session, query_embedding: List[float], n_items: int = 10):
    """
    Returns a list of tuples: (Ticket, score)
    score = inner_product between ticket.embedding and query_embedding
    """
    # Compute the scores
    raw_score = Ticket.embedding.max_inner_product(query_embedding) # Negative IP

    # flip sign and normalize to [0,1]
    score_expr = ((-raw_score + 1) / 2).label("score")

    # Query both the Ticket and the score
    results = (
        db.query(Ticket, score_expr)
        .order_by(score_expr.desc())
        .limit(n_items)
        .all()
    )

    # `results` is now a list of tuples: (Ticket, score)
    return results


