"""
Configuration Module

This module centralizes all application settings and paths.
"""
import os
import logging
import sys

DEBUG = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensure logs go to stdout
)

# --- Core Path Definitions ---

# BASE_DIR: The absolute path to the 'src' directory (where this file lives)
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR: The absolute path to the 'data' directory (one level up from 'src')
DATA_DIR: str = os.path.join(BASE_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# --- Model Configuration ---

# Load the name of the SentenceTransformer model to use from the environment so it can be changed dynamically
PRETRAINED_MODEL: str = str(os.getenv("PRETRAINED_SEARCH_MODEL"))

# --- File Paths (Derived from Model) ---

# MODEL_DATA_DIR: A dedicated directory to store artifacts for the *specific*
# model being used
MODEL_DATA_DIR: str = os.path.join(DATA_DIR, PRETRAINED_MODEL)
os.makedirs(MODEL_DATA_DIR, exist_ok=True)

CACHE_MODEL_DATA_DIR: str = os.path.join(MODEL_DATA_DIR, "cache") # path to the cached model
os.makedirs(CACHE_MODEL_DATA_DIR, exist_ok=True)

# INDEX_PATH: The file path where the trained FAISS index will be saved.
INDEX_PATH: str = os.path.join(MODEL_DATA_DIR, "index.faiss")

# --- Database and Source Data ---

# DB_PATH: The file path for the SQLite database.
DB_PATH: str = os.path.join(DATA_DIR, 'tickets.db')

# --- Default Configurations ---

# BATCH_PROCESS_INTERVAL: The time (in seconds) the background thread
# waits before checking for new index updates.
BATCH_PROCESS_INTERVAL: int = int(os.getenv("BATCH_PROCESS_INTERVAL"))