# Python built-in libraries
import logging
import time
import csv
import io

# External libraries
from fastapi import FastAPI, HTTPException, UploadFile, File
from redis import Redis
from rq import Queue

# Shared imports
import config
from preprocessing import clean_text
from embedding import get_embedding
from database import get_db, db_search_tickets, db_delete_ticket, db_start

# Module specific imports
from models import SearchRequest, TicketRequest, SearchHit, SearchResult, TicketResponse

logger = logging.getLogger("uvicorn")

# Connect to Redis
redis_conn = Redis(host=config.REDIS_HOST,
                   port=config.REDIS_PORT,
                   password=config.REDIS_PASSWORD)
# Create RQ queue
q = Queue(config.REDIS_QUEUE, connection=redis_conn)
logger.info("Redis Queue created.")

app = FastAPI(
    title="API Search Service",
    description="An API Search Engine to do semantic search on a database.",
    openapi_url="/api/openapi.json",  
    docs_url="/api/docs",             
)

@app.on_event("startup")
def startup():
    db_start()

@app.post("/api/search")
def api_search(req: SearchRequest):
    logger.info(f"Search for items={req.n_items}, query={req.query}")
    
    total_start = time.perf_counter()  # start total timer

    # Step 1: generate embedding
    query_embedding = get_embedding(clean_text(req.query))
    
    # Step 2: DB search
    search_start = time.perf_counter()

    with get_db() as db: 
        results = db_search_tickets(db, query_embedding, req.n_items)

    search_time = time.perf_counter() - search_start

    # Step 3: Convert results
    hits = [SearchHit(ticket=TicketResponse.model_validate(t), score=s) for t, s in results]
    total_time = time.perf_counter() - total_start

    return {
        "data": SearchResult(
            items=hits,
            count=len(hits),
            search_time=search_time,
            total_time=total_time
        )
    }

@app.post("/api/ticket")
def api_add_ticket(ticket: TicketRequest):
    logger.info("New ticket added.")

    # Queue the ticket (to also calculate the embeddings)
    job = q.enqueue("worker.dispatch", "add_ticket", ticket.model_dump())

    return {
        "message": "Ticket queued for creation",
        "data": {"job_id": job.id}
    }

@app.delete("/api/ticket/{ticket_id}")
def api_delete_ticket(ticket_id: int):
    # Delete directly from the DB both the ticket and the embeddings
    logger.info(f"Deleting ticket {ticket_id}")

    with get_db() as db:
        success = db_delete_ticket(db, ticket_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {
        "message": "Ticket deleted successfully",
        "data": {"ticket_id": ticket_id}
    }

@app.put("/api/ticket/{ticket_id}")
def api_update_ticket(ticket_id: int, ticket: TicketRequest):
    logger.info(f"Updating ticket: {ticket_id}")

    job = q.enqueue("worker.dispatch", "update_ticket", ticket_id, ticket.model_dump())

    return {
        "message": "Ticket queued for update",
        "data": {
            "ticket_id": ticket_id,
            "job_id": job.id
        }
    }

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    logger.info(f"Processing file: {file.filename}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    # Read file into memory
    content = await file.read()
    decoded = content.decode("utf-8")
    csv_file = io.StringIO(decoded)

    # CSV Check
    try:
        reader = csv.DictReader(csv_file)
        # Ensure headers exist
        if not reader.fieldnames:
            raise ValueError("CSV file has no headers")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    batch = []
    job_ids = []

    # Start reading
    for row in reader:
        ticket_data = {
            "short_description": row.get("short_description", "(No title)"),
            "category": row.get("category", "N/D"),
            "subcategory": row.get("subcategory", "N/D"),
            "content": row.get("content", "N/D"),
            "software": row.get("software", "N/D"),
        }

        batch.append(ticket_data)

        # Flush batch
        if len(batch) >= config.BATCH_SIZE:
            job = q.enqueue("worker.dispatch", "add_tickets_batch", batch)
            job_ids.append(job.id)
            batch = []

    # Flush remaining tickets
    if batch:
        job = q.enqueue("worker.dispatch", "add_tickets_batch", batch)
        job_ids.append(job.id)

    return {
        "message": f"{len(job_ids)} tickets queued for creation",
        "data": {"job_ids": job_ids}
    }