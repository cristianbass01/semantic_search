import os

# Sentence Transformer model configuration
HF_EMBEDDINGS_HOST = os.getenv("HF_EMBEDDINGS_HOST", "embeddings")
HF_EMBEDDINGS_PORT = os.getenv("HF_EMBEDDINGS_PORT", "5000")
HF_EMBEDDINGS_BASE_URL =  f"http://{HF_EMBEDDINGS_HOST}:{HF_EMBEDDINGS_PORT}"

HF_EMBEDDINGS_URL = f"{HF_EMBEDDINGS_BASE_URL}/embed"
HF_EMBEDDINGS_READY_URL = f"{HF_EMBEDDINGS_BASE_URL}/health"

BATCH_SIZE = os.getenv("BATCH_SIZE", 32)

# Postgres configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB", "tickets-db")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "ticket_queue")