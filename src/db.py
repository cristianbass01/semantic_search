"""
Database Utility Module (db_utils)

This module handles all database interactions for the application,
using a SQLite database. It includes functions for:
- Initializing the database from a CSV file.
- Performing CRUD (Create, Read, Update, Delete) operations on tickets.
- Efficiently batch-reading tickets for memory-intensive tasks.
"""

import sqlite3
import logging
from typing import List, Dict, Any, Tuple, Union
from . import config

# Get a logger instance for this module
log = logging.getLogger(__name__)

# Load constants from the central config file
DB_PATH = config.DB_PATH
    
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