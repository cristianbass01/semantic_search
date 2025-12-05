import os

# Custom code
from src import config as base_config

# --- Load necessary constants from base_config ---

PRETRAINED_MODEL: str = base_config.PRETRAINED_MODEL # the SentenceTransformer model to use

INDEX_PATH: str = base_config.INDEX_PATH # The file path where the trained FAISS index will be saved.

DB_PATH: str = base_config.DB_PATH # The file path for the SQLite database.

CACHE_MODEL_DATA_DIR: str = base_config.CACHE_MODEL_DATA_DIR # The path to the cached model

# --- Core Path Definitions ---

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__)) # Current Path

INIT_DATA_DIR: str = os.path.join(BASE_DIR, "init_data") # Path where to store initialization data

INIT_MODEL_DATA_DIR: str = os.path.join(INIT_DATA_DIR, PRETRAINED_MODEL) # Path where to store model specific initialization data
os.makedirs(INIT_MODEL_DATA_DIR, exist_ok=True)

# --- File Paths to load data and cache embeddings ---

EMBEDDING_PATH: str = os.path.join(INIT_MODEL_DATA_DIR, "embeddings.pkl") # The file path for the pickled embeddings and their IDs.

CSV_PATH: str = os.path.join(INIT_DATA_DIR, os.getenv("CSV_FILE")) # The path to the source CSV file used to populate the database.

# --- DEFAULT CONFIGURATIONS ---

DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE")) # The number of rows to process at a time when