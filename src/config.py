"""
Configuration Module

This module centralizes all application settings and paths.
"""

import os

# --- Core Path Definitions ---

# BASE_DIR: The absolute path to the 'src' directory (where this file lives)
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR: The absolute path to the 'data' directory (one level up from 'src')
DATA_DIR: str = os.path.join(BASE_DIR, '..', 'data')


# --- Model Configuration ---

# The name of the SentenceTransformer model to use.
# 'paraphrase-multilingual-mpnet-base-v2' is a strong multilingual model.
# 'multi-qa-mpnet-base-dot-v1' is a good alternative tuned for QA.
PRETRAINED_MODEL: str = 'paraphrase-multilingual-mpnet-base-v2'


# --- File Paths (Derived from Model) ---

# MODEL_DATA_DIR: A dedicated directory to store artifacts for the *specific*
# model being used (e.g., /data/paraphrase-multilingual-mpnet-base-v2)
MODEL_DATA_DIR: str = os.path.join(DATA_DIR, PRETRAINED_MODEL)

# INDEX_PATH: The file path where the trained FAISS index will be saved.
INDEX_PATH: str = os.path.join(MODEL_DATA_DIR, "index.faiss")

# EMBEDDING_PATH: The file path for the pickled embeddings and their IDs.
# This is used as a cache to speed up index re-building.
EMBEDDING_PATH: str = os.path.join(MODEL_DATA_DIR, "embeddings.pkl")


# --- Database and Source Data ---

# DB_PATH: The file path for the SQLite database.
DB_PATH: str = os.path.join(DATA_DIR, 'tickets.db')

# CSV_PATH: The path to the source CSV file used to populate the database.
CSV_PATH: str = os.path.join(DATA_DIR, "synthetic-it-call-center-tickets.csv")

# DEFAULT_BATCH_SIZE: The number of rows to process at a time when
# reading from the CSV or fetching all tickets.
DEFAULT_BATCH_SIZE: int = 10000


# --- Application Tunables ---

# BATCH_PROCESS_INTERVAL: The time (in seconds) the background thread
# waits before checking for new index updates. (10 minutes)
BATCH_PROCESS_INTERVAL: int = 10 * 60


# --- Runtime Initialization ---

# Ensure the directory for storing model-specific files exists.
os.makedirs(MODEL_DATA_DIR, exist_ok=True)