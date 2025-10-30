"""
Database Utility Module (db_utils)

This module handles all database interactions for the application,
using a SQLite database. It includes functions for:
- Initializing the database from a CSV file.
- Performing CRUD (Create, Read, Update, Delete) operations on tickets.
- Efficiently batch-reading tickets for memory-intensive tasks.
"""

import os
import sqlite3
import pandas as pd
import logging
from typing import Generator, List, Dict, Any, Tuple, Union
from . import config

# Get a logger instance for this module
log = logging.getLogger(__name__)

# Load constants from the central config file
DB_PATH = config.DB_PATH
CSV_PATH = config.CSV_PATH
DEFAULT_BATCH_SIZE = config.DEFAULT_BATCH_SIZE

def init_db():
    """Initializes the database.

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
                log.info("Database created. Table 'tickets' ready.")
            else:
                log.info("Database found. Table 'tickets' verified.")
                
    except sqlite3.Error as e:
        log.error(f"Error during database table initialization: {e}")
        return  # Exit if table creation fails

    # If the database is new, populate it from the CSV
    if not db_existed_before:
        log.info(f"New database detected. Starting data load from '{CSV_PATH}'...")
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
        except Exception as e:
            log.error(f"CRITICAL: An error occurred during CSV reading: {e}. DB is empty.")


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
    
def add_ticket(ticket_data: Dict[str, Any]) -> Tuple[bool, Union[int, str]]:
    """Adds a single new ticket to the database.

    Args:
        ticket_data: A dictionary containing the ticket data.
            Expected keys: 'short_description', 'content', 'category',
            'subcategory', 'software'.

    Returns:
        A tuple (success, result):
        - (True, new_ticket_id) on success.
        - (False, error_message) on failure.
    """
    insert_query = """
    INSERT INTO tickets (short_description, content, category, subcategory, software)
    VALUES (?, ?, ?, ?, ?);
    """
    data_tuple = (
        ticket_data.get('short_description'),
        ticket_data.get('content'),
        ticket_data.get('category'),
        ticket_data.get('subcategory'),
        ticket_data.get('software')
    )
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(insert_query, data_tuple)
            new_id = cursor.lastrowid  # Get the ID of the newly inserted row
            conn.commit()
            return True, new_id  # Return success and the new ID
    except sqlite3.Error as e:
        log.error(f"Error inserting ticket: {e}")
        return False, str(e)  # Return failure and the error message

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
    
def get_ticket_by_id(ticket_id: int) -> Union[Dict[str, Any], None]:
    """Fetches a single ticket by its ID.

    Args:
        ticket_id: The unique ID of the ticket to retrieve.

    Returns:
        A dictionary containing the ticket data if found, else None.
    """
    select_query = "SELECT * FROM tickets WHERE id = ?;"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row  # Get results as dictionaries
            cursor = conn.cursor()
            cursor.execute(select_query, (ticket_id,))
            result_row = cursor.fetchone()
            return dict(result_row) if result_row else None
    except sqlite3.Error as e:
        log.error(f"Error querying ticket ID {ticket_id}: {e}")
        return None
        
def get_tickets_by_ids(ticket_ids: List[int]) -> List[Dict[str, Any]]:
    """Fetches multiple tickets from the database by their IDs.

    Args:
        ticket_ids: A list of ticket IDs to retrieve.

    Returns:
        A list of dictionaries, one for each *found* ticket,
        sorted in the same order as the input `ticket_ids`.
    """
    if not ticket_ids:
        return []
        
    # Create dynamic '?' placeholders for the IN clause
    placeholders = ','.join(['?'] * len(ticket_ids))
    select_query = f"SELECT * FROM tickets WHERE id IN ({placeholders});"
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(select_query, ticket_ids)
            results = [dict(row) for row in cursor.fetchall()]
            
            # Re-sort the results to match the original 'ticket_ids' order
            ordered_results = sorted(results, key=lambda r: ticket_ids.index(r['id']))
            return ordered_results
    except sqlite3.Error as e:
        log.error(f"Error querying multiple ticket IDs: {e}")
        return []
    
    
def update_ticket(ticket_id: int, ticket_data: Dict[str, Any]) -> Tuple[bool, Union[str, None]]:
    """Updates a ticket in the database with new data.

    This function dynamically builds an UPDATE query to change only the
    fields provided in the `ticket_data` dictionary.

    Args:
        ticket_id: The ID of the ticket to update.
        ticket_data: A dictionary where keys are the column names
            (e.g., 'content', 'category') and values are the new data.

    Returns:
        A tuple (success, message):
        - (True, None) on success.
        - (False, error_message) on failure (e.g., "Ticket not found").
    """
    
    if not ticket_data:
        log.warning("Empty update data provided, no operation performed.")
        return True, None  # Success (nothing to do)

    # Dynamically and safely build the SET clause
    set_clauses = []
    parameters = []
    
    # Whitelist of fields that are allowed to be updated
    allowed_fields = ['short_description', 'content', 'category', 'subcategory', 'software']
    
    for key, value in ticket_data.items():
        if key in allowed_fields:
            set_clauses.append(f"{key} = ?")
            parameters.append(value)
        else:
            log.warning(f"Field '{key}' cannot be updated and will be ignored.")

    if not set_clauses:
        msg = "No valid fields provided for update."
        log.error(msg)
        return False, msg

    # Add the ticket_id to the end of the parameters list for the WHERE clause
    parameters.append(ticket_id)
    
    # Build the final query, e.g.:
    # "UPDATE tickets SET content = ?, category = ? WHERE id = ?"
    update_query = f"""
    UPDATE tickets
    SET {', '.join(set_clauses)}
    WHERE id = ?;
    """
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(update_query, parameters)
            
            # Check if any row was actually affected
            if cursor.rowcount == 0:
                msg = f"Update failed: No ticket found with ID {ticket_id}."
                log.error(msg)
                return False, "Ticket not found"
                
            conn.commit()
            return True, None  # Success
            
    except sqlite3.Error as e:
        log.error(f"Error updating ticket ID {ticket_id}: {e}")
        return False, str(e)  # Failure

def delete_tickets_by_ids(ticket_ids: List[int]) -> Tuple[bool, Union[int, str]]:
    """Deletes multiple tickets from the database in a single transaction.

    Args:
        ticket_ids: A list of ticket IDs to delete.

    Returns:
        A tuple (success, result):
        - (True, rows_deleted_count) on success.
        - (False, error_message) on failure.
    """
    
    if not ticket_ids:
        log.warning("No ticket IDs provided for deletion.")
        return True, 0  # Success (nothing to do), 0 rows deleted

    try:
        # Create dynamic '?' placeholders for the IN clause
        placeholders = ','.join(['?'] * len(ticket_ids))
        delete_query = f"DELETE FROM tickets WHERE id IN ({placeholders});"
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(delete_query, ticket_ids)
            
            rows_deleted = cursor.rowcount  # Number of rows actually deleted
            
            conn.commit()
            log.info(f"Successfully deleted {rows_deleted} rows.")
            return True, rows_deleted
            
    except sqlite3.Error as e:
        log.error(f"Error during bulk deletion: {e}")
        return False, str(e)