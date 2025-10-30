"""
Semantic Search Utility Module (search_utils)

This module defines the core SemanticSearcher class, which handles:
- Loading and managing the SentenceTransformer model.
- Building, loading and saving the vector index.
- Preprocessing and cleaning text data.
- Running a background thread to safely batch-update the index.
- Providing thread-safe functions to add the dirty IDs to the buffer.
"""

import os
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import faiss
from . import db_utils
from . import config

import threading
import time
import atexit
from typing import List, Dict, Any, Set

# Get a logger instance for this module
log = logging.getLogger(__name__)

# --- Load Constants from Config ---
PRETRAINED_MODEL = config.PRETRAINED_MODEL
INDEX_PATH = config.INDEX_PATH
EMBEDDING_PATH = config.EMBEDDING_PATH
BATCH_PROCESS_INTERVAL = config.BATCH_PROCESS_INTERVAL

#################################################
#              Preprocessing Functions          #
#################################################

def clean_series(series: pd.Series) -> pd.Series:
    """Cleans a pandas Series of text data.

    This function is optimized for semantic models:
    - Removes HTML tags.
    - Normalizes all whitespace (newlines, tabs, etc.) to a single space.
    - Trims leading/trailing whitespace.
    - Keeps punctuation and casing, as they are important for semantic meaning.
    
    Args:
        series: The input pandas Series containing text.

    Returns:
        A new pandas Series with the cleaned text.
    """
    s_cleaned = series.fillna('').astype(str)
    s_cleaned = s_cleaned.str.replace(r'<[^>]+>', ' ', regex=True)  # Remove HTML
    s_cleaned = s_cleaned.str.replace(r'\s+', ' ', regex=True)     # Normalize whitespace
    s_cleaned = s_cleaned.str.strip()                             # Trim
    return s_cleaned

def preprocess_batch(df_batch: pd.DataFrame) -> pd.Series:
    """Converts a DataFrame of ticket data into a single text Series for embedding.

    This function formats multiple columns into a single descriptive string
    for each ticket, which is then fed to the embedding model.

    Args:
        df_batch: A DataFrame containing ticket data. Must include 'id'
            and is expected to have 'short_description', 'content', 'category',
            'subcategory', and 'software/system'.

    Returns:
        A pandas Series where each item is the formatted string
        ready for embedding.
    """
    
    batch_index = df_batch.index
    batch_columns = df_batch.columns

    def get_col_safe(col_name: str) -> pd.Series:
        """Helper to safely get a column or return an empty Series."""
        if col_name in batch_columns:
            return df_batch[col_name]
        else:
            # Return an empty Series with the same index
            return pd.Series(dtype=str).reindex(batch_index)

    # Clean each data column individually
    s_cat = clean_series(get_col_safe('category'))
    s_sub = clean_series(get_col_safe('subcategory'))
    s_desc = clean_series(get_col_safe('short_description'))
    s_cont = clean_series(get_col_safe('content'))
    s_sw = clean_series(get_col_safe('software/system'))

    # Combine into the final formatted string
    # This format is designed to give the model context
    final_text_series = (
        "Title: " + s_desc + " | " +
        "Category: " + s_cat + " " +  s_sub + " | " +
        "Software: " + s_sw + " | " +
        "Content: " + s_cont
    )
    
    return final_text_series

#################################################
#             Semantic Search Class             #
#################################################

class SemanticSearcher():
    """Manages the semantic search model, vector index, and update queue.

    This class encapsulates all logic for loading the model, building the
    FAISS index from the database, and handling real-time, thread-safe
    updates to the index.
    
    Attributes:
        model (SentenceTransformer): The loaded embedding model.
        index (faiss.IndexIDMap): The FAISS vector index.
        index_lock (threading.Lock): Lock protecting read/write access to self.index.
        queue_lock (threading.Lock): Lock protecting access to the update queues.
    """

    def __init__(self):
        """Initializes the model, index, and background update thread."""
        log.info("Initializing SemanticSearcher...")

        # --- Model and DB Initialization ---
        self.model: SentenceTransformer = SentenceTransformer(PRETRAINED_MODEL, device='cpu')
        log.info(f"Model {PRETRAINED_MODEL} loaded.")
        
        # Ensure database and tables exist before trying to read from them
        db_utils.init_db()

        # --- Index Initialization Strategy ---
        # We try to load the index in this order:
        # 1. Load the pre-built FAISS index file (fastest)
        # 2. Load pre-computed embeddings and build the index in memory (fast)
        # 3. Re-compute all embeddings from the database (slowest)
        
        if os.path.exists(INDEX_PATH):
            # 1. Load the fully built index from disk
            log.info("Loading pre-built FAISS index from disk...")
            self.index: faiss.IndexIDMap = faiss.read_index(INDEX_PATH)
        
        else:
            # Index file not found, try to build it
            if os.path.exists(EMBEDDING_PATH):
                # 2. Load pre-computed embeddings from pickle file
                log.info("Loading pre-computed embeddings from disk...")
                with open(EMBEDDING_PATH, "rb") as fIn:
                    embedding_data = pickle.load(fIn)
                    ids = embedding_data['ids']
                    embeddings = embedding_data['embeddings']

            else:
                # 3. Re-compute all embeddings from the database
                log.info("No index or embeddings found. Recomputing from database...")
                ticket_generator = db_utils.get_all_tickets()

                all_ids_list = []
                all_embeddings_list = []
                total_tickets_embedded = 0

                for batch_df in ticket_generator:
                    log.info(f"Embedding batch of {len(batch_df)} tickets. Total: {total_tickets_embedded}")
                    
                    batch_ids = batch_df['id'].values
                    batch_sentences = preprocess_batch(batch_df)

                    batch_embeddings = self.model.encode(
                        batch_sentences.tolist(), 
                        show_progress_bar=False, 
                        convert_to_numpy=True, 
                        normalize_embeddings=True  # Normalize for Cosine Similarity
                        )
                    
                    total_tickets_embedded += len(batch_df)
                    all_ids_list.append(batch_ids)
                    all_embeddings_list.append(batch_embeddings)

                if not all_embeddings_list:
                    # Handle case where database is empty
                    log.warning("No data found in database to build index.")
                    d = self.model.get_sentence_embedding_dimension()
                    self.index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
                    return  # Exit init early
                
                embeddings = np.concatenate(all_embeddings_list, axis=0)
                ids = np.concatenate(all_ids_list, axis=0).astype('int64')

                # Save the computed embeddings for the next startup
                data_to_save = {'ids': ids, 'embeddings': embeddings}
                with open(EMBEDDING_PATH, "wb") as fOut:
                    pickle.dump(data_to_save, fOut)
                log.info(f"Saved {len(ids)} new embeddings to {EMBEDDING_PATH}")
            
            # --- Build and Save FAISS Index from Embeddings ---
            log.info(f"Building FAISS index with {len(ids)} vectors...")
            embedding_size = embeddings.shape[1]
            # Use IndexFlatIP (Inner Product) because embeddings are normalized
            # (Normalized Inner Product == Cosine Similarity)
            index_flat = faiss.IndexFlatIP(embedding_size)
            self.index = faiss.IndexIDMap(index_flat)
            self.index.add_with_ids(embeddings, ids)

            # Save the new index to disk for the next startup
            faiss.write_index(self.index, INDEX_PATH)
            log.info(f"Index built and saved to {INDEX_PATH}.")
        
        # --- Threading and Concurrency Setup ---
        self.index_lock: threading.Lock = threading.Lock()  # Protects self.index
        self.queue_lock: threading.Lock = threading.Lock()  # Protects dirty sets
        
        self.dirty_ids_to_add: Set[int] = set()
        self.dirty_ids_to_remove: Set[int] = set()
        
        self.shutdown_event: threading.Event = threading.Event()
        self.work_event: threading.Event = threading.Event()
        
        # Start the background thread to process index updates
        self.batch_processor_thread: threading.Thread = threading.Thread(
            target=self._batch_processor_loop, 
            daemon=True  # Thread will exit if the main program exits
        )
        self.batch_processor_thread.start()
        
        # Register the graceful shutdown function
        atexit.register(self.shutdown)
        log.info("SemanticSearcher initialized and background thread started.")

    def _batch_processor_loop(self):
        """The main loop for the background update thread.
        
        This loop waits for a signal (or a timeout) and then processes
        any pending updates to the FAISS index.
        """
        log.info(f"Batch processor thread started. Checking every {BATCH_PROCESS_INTERVAL}s.")
        while not self.shutdown_event.is_set():
            try:
                # Wait for the timeout or for self.work_event.set() to be called
                self.work_event.wait(timeout=BATCH_PROCESS_INTERVAL)
                
                # Clear the event so we wait again next time
                self.work_event.clear()
                
                if self.shutdown_event.is_set():
                    break
                    
                self._process_batch()
                
            except Exception as e:
                log.error(f"Error in batch processor loop: {e}", exc_info=True)
                # Don't spin-lock on a persistent error; wait a bit
                time.sleep(60)

    def _process_batch(self):
        """Processes all queued additions and deletions to the FAISS index."""
        ids_to_remove_list: List[int] = []
        ids_to_add_list: List[int] = []

        # Safely copy and clear the update queues
        with self.queue_lock:
            if not self.dirty_ids_to_add and not self.dirty_ids_to_remove:
                return  # Nothing to do
            
            ids_to_remove_list = list(self.dirty_ids_to_remove)
            ids_to_add_list = list(self.dirty_ids_to_add)
            self.dirty_ids_to_remove.clear()
            self.dirty_ids_to_add.clear()
            
        log.info(f"Processing batch: {len(ids_to_remove_list)} removals, {len(ids_to_add_list)} additions/updates.")
        
        new_embeddings = []
        new_ids = np.array([], dtype='int64')

        # Fetch data and compute embeddings for new/updated items
        if ids_to_add_list:
            tickets_data = db_utils.get_tickets_by_ids(ids_to_add_list)
            
            if tickets_data:
                df = pd.DataFrame(tickets_data)
                sentences = preprocess_batch(df)
                new_embeddings = self.model.encode(
                    sentences.tolist(),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                new_ids = df['id'].values.astype('int64')
            else:
                log.warning(f"Requested to add {len(ids_to_add_list)} IDs, but none were found in DB.")

        # Atomically update the FAISS index
        log.info("Acquiring index_lock for update...")
        with self.index_lock:
            # We must remove before adding to handle update operations
            if ids_to_remove_list:
                ids_to_remove_np = np.array(ids_to_remove_list, dtype='int64')
                # Use IDSelectorBatch for efficient removal
                selector = faiss.IDSelectorBatch(ids_to_remove_np)
                removed_count = self.index.remove_ids(selector)
                log.info(f"Removed {removed_count} embeddings from index.")

            # Add new and updated embeddings
            if len(new_ids) > 0:
                self.index.add_with_ids(new_embeddings, new_ids)
                log.info(f"Added {len(new_ids)} new embeddings to index.")
        
        log.info("Released index_lock.")
        
        # Persist the updated index to disk
        self.save_index()
        log.info("Batch update complete. Index saved to disk.")

    def save_index(self):
        """Saves the current FAISS index to disk in a thread-safe way."""
        with self.index_lock:
            try:
                log.info(f"Saving index to {INDEX_PATH}...")
                faiss.write_index(self.index, INDEX_PATH)
            except Exception as e:
                log.error(f"CRITICAL: Failed to save index to disk: {e}", exc_info=True)

    def shutdown(self):
        """Performs a graceful shutdown of the background thread."""
        log.info("Shutdown requested for SemanticSearcher...")
        
        # Signal the thread to stop
        self.shutdown_event.set()
        self.work_event.set()  # Wake it up if it's waiting
        
        # Wait for the thread to finish
        self.batch_processor_thread.join(timeout=10)
        log.info("Batch processor thread stopped.")
        
        # Process any final items that were queued during shutdown
        log.info("Running final batch process before exit...")
        self._process_batch()
        
        log.info("SemanticSearcher shut down cleanly.")

    def search(self, query: str, top_k_hits: int) -> List[Dict[str, Any]]:
        """Performs a thread-safe semantic search.

        Args:
            query: The user's search query string.
            top_k_hits: The maximum number of results to return.

        Returns:
            A list of result dictionaries, sorted by score. Each dictionary
            contains the 'id', 'score', and the full 'ticket' data.
        """
        # Clean the query using the exact same pipeline as the documents
        clean_query = clean_series(pd.Series([query])).iloc[0]
        
        # Encode the query
        query_embedding = self.model.encode(
            [clean_query],
            show_progress_bar=False, 
            convert_to_numpy=True, 
            normalize_embeddings=True  # Must match document normalization
        )

        # Perform the search in a thread-safe way
        with self.index_lock:
            distances, corpus_ids = self.index.search(query_embedding, top_k_hits)

        # Post-process the results
        hits = []
        if len(corpus_ids[0]) > 0:
            # Filter out any invalid IDs (-1)
            hit_ids = [int(id) for id in corpus_ids[0] if int(id) != -1]
            if not hit_ids:
                return []
                
            # Get the full ticket data from the database
            db_results = db_utils.get_tickets_by_ids(hit_ids)
            
            # Create a score lookup map
            score_map = {int(id): float(score) for id, score in zip(corpus_ids[0], distances[0])}
            
            # Combine DB data with similarity scores
            for ticket in db_results:
                ticket_id = ticket['id']
                hits.append({
                    "id": ticket_id,
                    "score": score_map.get(ticket_id, 0.0),
                    "ticket": ticket  # Include the full ticket object
                })
        
        # Sort results by score (highest is best for Cosine Similarity)
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        return hits

    def add(self, ticket_id: int):
        """Queues a new ticket to be added to the index.

        Args:
            ticket_id: The ID of the ticket to add.
        """
        log.info(f"Queuing ADD for ticket: {ticket_id}")
        with self.queue_lock:
            self.dirty_ids_to_add.add(ticket_id)
        self.work_event.set()  # Wake up the worker thread

    def remove(self, ticket_id: int):
        """Queues a ticket to be removed from the index.

        Args:
            ticket_id: The ID of the ticket to remove.
        """
        log.info(f"Queuing REMOVE for ticket: {ticket_id}")
        with self.queue_lock:
            self.dirty_ids_to_remove.add(ticket_id)
            # If it was also queued for adding, discard that action
            self.dirty_ids_to_add.discard(ticket_id)
        self.work_event.set()  # Wake up the worker thread

    def update(self, ticket_id: int):
        """Queues a ticket to be updated in the index.

        This is handled by removing the old vector and adding a new one.
        
        Args:
            ticket_id: The ID of the ticket to update.
        """
        log.info(f"Queuing UPDATE for ticket: {ticket_id}")
        with self.queue_lock:
            self.dirty_ids_to_remove.add(ticket_id)
            self.dirty_ids_to_add.add(ticket_id)
        self.work_event.set()  # Wake up the worker thread