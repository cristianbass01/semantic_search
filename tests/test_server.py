import pytest
from src.server import create_app

@pytest.fixture
def app():
    """
    Create and configure a new app instance for each test.
    """
    app = create_app()
    app.config.update({"TESTING": True})
    yield app

@pytest.fixture
def client(app):
    """
    A test client for the app to send fake HTTP requests.
    """
    return app.test_client()

def test_home_route(client):
    """
    Tests that the homepage loads.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.data

def test_api_search(client, mocker):
    """
    Tests the /api/search endpoint.
    """
    # Create a fake search result
    fake_results = [
        {"id": 123, "score": 0.99, "ticket": {"short_description": "Fake result"}}
    ]
    
    # Patch the search on the app instance
    mocker.patch.object(
        client.application.search_service, 
        'search', 
        return_value=fake_results
    )
    
    # Call the API endpoint
    response = client.post("/api/search", json={
        "query": "test query",
        "n_items": 1
    })
    
    # Check the results
    assert response.status_code == 200
    json_data = response.json
    assert len(json_data) == 1
    assert json_data[0]["id"] == 123

def test_api_add_ticket(client, mocker):
    """
    Tests the POST /api/ticket endpoint.
    """
    # Patch db_utils
    mocker.patch('src.server.db_utils.add_ticket', return_value=(True, 999))
    
    # Patch the add method on the app instance
    mock_search_add = mocker.patch.object(
        client.application.search_service, 
        'add'
    )
    
    # 3. Call the API
    ticket_data = {"short_description": "New test ticket"}
    response = client.post("/api/ticket", json=ticket_data)
    
    # 4. Check the results
    assert response.status_code == 201 # 201 Created
    assert response.json["id"] == 999
    
    # Verify the search service was cued to add the *new* ID
    mock_search_add.assert_called_once_with(999)