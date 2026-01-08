from database import get_db, db_add_ticket, db_update_ticket, db_start, db_add_tickets_batch
from preprocessing import preprocess_ticket
from embedding import get_embedding, get_embeddings_batch
import config

from rq import Worker, Queue
from redis import Redis
import logging

log = logging.getLogger()

redis_conn = Redis(host=config.REDIS_HOST,
                       port=config.REDIS_PORT,
                       password=config.REDIS_PASSWORD)
    
q = Queue(config.REDIS_QUEUE, connection=redis_conn)

def add_ticket(ticket_dict: dict):
    log.info("New ticket received.")

    try:
        ticket_text = preprocess_ticket(ticket_dict)

        embedding = get_embedding(ticket_text)

        with get_db() as db:
            db_add_ticket(db, ticket_dict, embedding)

    except Exception as e:
        log.error(f"Fail to add ticket: {e}")
        raise

def add_tickets_batch(tickets: list[dict]):
    log.info(f"Adding batch of {len(tickets)} tickets")

    try:
        texts = [preprocess_ticket(t) for t in tickets]

        embeddings = get_embeddings_batch(texts)

        with get_db() as db:
            db_add_tickets_batch(db, tickets, embeddings)

    except Exception as e:
        log.error(f"Batch add failed: {e}")
        raise
    

def update_ticket(ticket_id, ticket_dict: dict):
    log.info(f"Updating ticket: {ticket_id}")

    try:
        ticket_text = preprocess_ticket(ticket_dict)

        embedding = get_embedding(ticket_text)

        with get_db() as db:
            db_update_ticket(db, ticket_id, ticket_dict, embedding)
            
    except Exception as e:
        log.error(f"Fail to update ticket: {e}")
        raise

TASK_MAP = {
    "add_ticket": add_ticket,
    "update_ticket": update_ticket,
    "add_tickets_batch": add_tickets_batch
}

def dispatch(task_name, *args, **kwargs):
    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name}")

    return TASK_MAP[task_name](*args, **kwargs)

if __name__ == "__main__":
    db_start()

    worker = Worker([q], connection=redis_conn)
    worker.work()