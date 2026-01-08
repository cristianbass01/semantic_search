import requests
import math
import config
import time
import logging 

log = logging.getLogger("uvicorn")

def get_embedding(text: str):
    return get_embeddings_batch(text)[0]

def get_embeddings_batch(batch: str):
    log.debug(f"Embedding batch of size {len(batch)}")
    while not embedding_is_ready():
        time.sleep(2)

    payload = {
        "inputs": batch,
        "normalize": True,
        "truncate": True,
        "truncation_direction": "right"
        }
    response = requests.post(config.HF_EMBEDDINGS_URL, json=payload, timeout=30)
    response.raise_for_status()
    
    embeddings = response.json()
    
    return embeddings 

def get_embedding_dim(probe_text: str = "embedding-dimension-probe") -> int:
    embedding = get_embedding(probe_text)
    return len(embedding)

def embedding_is_ready():
    try:
        response = requests.get(config.HF_EMBEDDINGS_READY_URL)
    except Exception as e:
        log.warning(f"Embedding not ready: {e}")
        return False

    if response.status_code == 200:
        return True
    else:
        log.warning("Embedding not ready.")
        return False 
    
