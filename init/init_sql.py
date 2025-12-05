import os
import sqlite3
import pandas as pd
import logging
from typing import Generator

# Custom code
import config

# Get a logger instance for this module
log = logging.getLogger(__name__)

# Load constants from the central config file
DB_PATH = config.DB_PATH
CSV_PATH = config.CSV_PATH
DEFAULT_BATCH_SIZE = config.DEFAULT_BATCH_SIZE

def init_db():
    """
    Initializes the database.

    1. Checks if the database file already exists.
    2. Creates the 'tickets' table if it doesn't exist.
    3. If the database was newly created, it populates it from the CSV file.
    """
    
    # Check if the DB file exists *before* connect() creates it.
    # This is crucial for detecting a fresh initialization.
    db_existed_before = os.path.exists(DB_PATH)

    create_table_query = """
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        short_description TEXT NOT NULL,
        content TEXT,
        category TEXT,
        subcategory TEXT,
        software TEXT
    );
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_query)
            if not db_existed_before:
                log.info(f"Database created at {DB_PATH}")
            else:
                log.info(f"Database found at {DB_PATH}")
                
    except sqlite3.Error as e:
        log.error(f"Error during database table initialization: {e}")
        return  # Exit if table creation fails

    # If the database is new, populate it from the CSV
    if not db_existed_before:
        log.info(f"New database detected. Starting data load from '{CSV_PATH}' with chunksize={DEFAULT_BATCH_SIZE}...")
        try:
            # Use a chunksize iterator to avoid loading the entire CSV into memory
            csv_iterator = pd.read_csv(CSV_PATH, chunksize=DEFAULT_BATCH_SIZE)
            
            for i, chunk_df in enumerate(csv_iterator):
                log.info(f"Processing CSV chunk {i+1} (rows: {len(chunk_df)})...")
                
                # Standardize column names (e.g., 'software/system' -> 'software')
                if 'software/system' in chunk_df.columns:
                    chunk_df.rename(columns={'software/system': 'software'}, inplace=True)

                # Insert the current chunk into the database
                success = add_tickets_from_dataframe(chunk_df)
                
                if not success:
                    log.error(f"Failed to insert chunk {i+1}. Stopping data load.")
                    break
            
            log.info("Data loading from CSV completed.")

        except FileNotFoundError:
            log.error(f"CRITICAL: CSV file not found at '{CSV_PATH}'. DB is empty.")
            raise FileNotFoundError({CSV_PATH})
        except Exception as e:
            log.error(f"CRITICAL: An error occurred during CSV reading: {e}. DB is empty.")
            raise Exception


def add_tickets_from_dataframe(df: pd.DataFrame) -> bool:
    """Adds tickets from a DataFrame to the database in a single transaction.

    This method is optimized for bulk inserts and is much faster
    than calling add_ticket() in a loop.

    Args:
        df: A DataFrame containing the tickets to add. Columns must
            match the 'tickets' table schema.

    Returns:
        True on success, False on failure.
    """
    
    try:
        # Define the exact column order required by the INSERT query
        db_columns = ['short_description', 'content', 'category', 'subcategory', 'software']
        df_ordered = df[db_columns]
        
        # Convert the DataFrame to a list of tuples for executemany()
        data_tuples = list(df_ordered.to_records(index=False))
        
    except KeyError as e:
        log.error(f"DataFrame is missing a required column: {e}. Insert failed.")
        return False
    
    num_inserted = len(data_tuples)
    if num_inserted == 0:
        log.warning("Empty DataFrame provided, no insert performed.")
        return True

    insert_query = """
    INSERT INTO tickets (short_description, content, category, subcategory, software)
    VALUES (?, ?, ?, ?, ?);
    """

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Execute the bulk insert in a single transaction
            cursor.executemany(insert_query, data_tuples)
            conn.commit()
            
            log.info(f"Successfully inserted {num_inserted} rows from DataFrame.")
            return True
            
    except sqlite3.Error as e:
        # The 'with' block automatically handles the ROLLBACK on error
        log.error(f"Error during bulk insert: {e}")
        return False
    
def get_all_tickets(batch_size: int = DEFAULT_BATCH_SIZE) -> Generator[pd.DataFrame, None, None]:
    """Fetches all tickets from the database as a DataFrame generator.

    This memory-efficient approach is ideal for large datasets, as it
    yields data in batches rather than loading everything at once.

    Args:
        batch_size: The number of tickets to fetch in each batch.
                    Defaults to DEFAULT_BATCH_SIZE.

    Yields:
        pd.DataFrame: A DataFrame containing a batch of tickets.
    """
    select_query = "SELECT * FROM tickets;"
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Use sqlite3.Row factory to get results as dictionaries
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(select_query)
            
            log.info(f"Starting batch retrieval (size: {batch_size})...")
            
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break  # No more rows left
                    
                # Convert the list of sqlite3.Row objects to a DataFrame
                batch_data = [dict(row) for row in rows]
                df_batch = pd.DataFrame(batch_data)
                
                yield df_batch  # Yield the current DataFrame batch
                
            log.info("Batch retrieval completed.")

    except sqlite3.Error as e:
        log.error(f"Error during batch ticket retrieval: {e}")
        # Return an empty generator on error
        return