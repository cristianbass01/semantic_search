import pytest
import sqlite3
from src import db_utils
from src import config
import pandas as pd

@pytest.fixture
def mock_db(mocker):
    """
    Create a fresh, in-memory database for each test.
    """
    # Use an in-memory database
    conn = sqlite3.connect(":memory:")
    
    # Mock the DB_PATH to return this connection so db_utils functions use it automatically.
    mocker.patch('src.db_utils.sqlite3.connect', return_value=conn)
    
    # Run the init_db logic
    db_utils.init_db()
    
    # Yield to run tests
    yield conn
    
    # Close the connection
    conn.close()


def test_add_and_get_ticket(mock_db):
    """
    Tests that we can add a ticket and then retrieve it.
    """
    ticket_data = {
        "short_description": "Test Ticket",
        "content": "This is a test.",
        "category": "Test",
        "subcategory": "Mocking",
        "software": "pytest"
    }
    
    # Add the ticket
    success, new_id = db_utils.add_ticket(ticket_data)
    
    assert success is True
    assert new_id == 1 # First ticket in the DB
    
    # Retrieve the ticket
    retrieved_ticket = db_utils.get_ticket_by_id(new_id)
    
    assert retrieved_ticket is not None
    assert retrieved_ticket['id'] == new_id
    assert retrieved_ticket['short_description'] == "Test Ticket"
    assert retrieved_ticket['content'] == "This is a test."

def test_get_tickets_by_ids(mock_db):
    """
    Tests retrieving multiple tickets.
    """
    # Add two tickets
    db_utils.add_ticket({"short_description": "Ticket 1"})
    db_utils.add_ticket({"short_description": "Ticket 2"})
    
    # Retrieve them
    tickets = db_utils.get_tickets_by_ids([1, 2])
    
    assert len(tickets) == 2
    assert tickets[0]['id'] == 1
    assert tickets[1]['id'] == 2
    assert tickets[0]['short_description'] == "Ticket 1"

def test_update_ticket(mock_db):
    """
    Tests the update_ticket function.
    """
    # Add a ticket
    success, new_id = db_utils.add_ticket({"short_description": "Original"})
    assert success is True
    
    # Update it
    update_data = {"short_description": "Updated", "category": "New Cat"}
    success, msg = db_utils.update_ticket(new_id, update_data)
    
    assert success is True
    assert msg is None
    
    # Verify the update
    retrieved = db_utils.get_ticket_by_id(new_id)
    assert retrieved['short_description'] == "Updated"
    assert retrieved['category'] == "New Cat"

def test_delete_ticket(mock_db):
    """
    Tests the delete_tickets_by_ids function.
    """
    # Add a ticket
    success, new_id = db_utils.add_ticket({"short_description": "To Delete"})
    assert db_utils.get_ticket_by_id(new_id) is not None
    
    # Delete it
    success, count = db_utils.delete_tickets_by_ids([new_id])
    assert success is True
    assert count == 1
    
    # Verify it's deleted
    assert db_utils.get_ticket_by_id(new_id) is None