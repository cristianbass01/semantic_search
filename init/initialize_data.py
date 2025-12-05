"""
Initialisation Module

Initializes the database, embedding cache and index is not present
"""

# Libraries
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import faiss

# Custom files
import config
from src.preprocessing import preprocess_batch
import init_sql

# Get a logger instance for this module
log = logging.getLogger(__name__)

# --- Load Constants from Config ---
PRETRAINED_MODEL = config.PRETRAINED_MODEL
INDEX_PATH = config.INDEX_PATH
EMBEDDING_PATH = config.EMBEDDING_PATH
CACHE_MODEL_DATA_DIR = config.CACHE_MODEL_DATA_DIR

log.info("Initializing SemanticSearcher...")

# --- Database Initialization
init_sql.init_db()

# --- Index Initialization Strategy ---
# We try to load the index in this order:
# 1. Load the pre-built FAISS index file (fastest)
# 2. Load pre-computed embeddings and build the index in memory (fast)
# 3. Re-compute all embeddings from the database (slowest)

if os.path.exists(INDEX_PATH):
    log.info("Index already initialized.")
else:
    log.info(f"No index found in {INDEX_PATH}. Recomputing from database...")

    existing_ids = set()
    all_ids_list = []
    all_embeddings_list = []

    # Index file not found, try to build it
    if os.path.exists(EMBEDDING_PATH):
        # 2. Load pre-computed embeddings from pickle file
        log.info(f"Loading pre-computed embeddings from {EMBEDDING_PATH}...")
        with open(EMBEDDING_PATH, "rb") as fIn:
            embedding_data = pickle.load(fIn)
            all_ids_list = [embedding_data['ids']]
            all_embeddings_list = [embedding_data['embeddings']]
            existing_ids = set(embedding_data['ids'])
    else:
        log.info(f"No embeddings found in {EMBEDDING_PATH}")

    # 3. Check if we need to embed other tickets
    ticket_generator = init_sql.get_all_tickets()
    
    # --- Model Instanciated ---
    model = None
    
    total_tickets_embedded = 0

    for batch_df in ticket_generator:
        log.info(f"Embedding batch of {len(batch_df)} tickets. Total: {total_tickets_embedded}")
        
        # Encode only not cached tickets
        missing_mask = ~batch_df["id"].isin(existing_ids)

        if any(missing_mask):
            # Initialize the model if not already done
            if model is None:
                model = SentenceTransformer(PRETRAINED_MODEL, cache_folder=CACHE_MODEL_DATA_DIR)
                log.info(f"Model loaded from {PRETRAINED_MODEL}.")

            batch_df = batch_df[missing_mask]

            # Preprocess tickets for embedding
            batch_ids = batch_df['id'].values
            batch_sentences = preprocess_batch(batch_df)

            # Create the embeddings using the model loaded
            batch_embeddings = model.encode(
                batch_sentences.tolist(), 
                show_progress_bar=False, 
                convert_to_numpy=True, 
                normalize_embeddings=True  # Normalize for Cosine Similarity
                )
            
            total_tickets_embedded += len(batch_df)
            all_ids_list.append(batch_ids)
            all_embeddings_list.append(batch_embeddings)
    
    embeddings = np.concatenate(all_embeddings_list, axis=0)
    ids = np.concatenate(all_ids_list, axis=0).astype('int64')

    if total_tickets_embedded > 0:
        # Save the computed embeddings for the next startup (if new embeddings have been created)
        data_to_save = {'ids': ids, 'embeddings': embeddings}
        with open(EMBEDDING_PATH, "wb") as fOut:
            pickle.dump(data_to_save, fOut)
        log.info(f"Saved {len(ids)} embeddings to {EMBEDDING_PATH}")
    
    # --- Build and Save FAISS Index from Embeddings ---
    log.info(f"Building FAISS index with {len(ids)} vectors...")
    embedding_size = embeddings.shape[1]
    # Use IndexFlatIP (Inner Product) because embeddings are normalized
    # (Normalized Inner Product == Cosine Similarity)
    index_flat = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIDMap(index_flat)
    index.add_with_ids(embeddings, ids)

    # Save the new index to disk for the next startup
    faiss.write_index(index, INDEX_PATH)
    log.info(f"Index built and saved to {INDEX_PATH}.")