"""
Main Flask Web Server (server.py)

This file serves as the main entry point for the web application.
It defines:
- The Flask application factory.
- All API endpoints for search, add, update, and delete.
- The development server entry point.
"""

import logging
from flask import Flask, request, jsonify, render_template, Response
from .search_utils import SemanticSearcher
from . import db_utils 
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_app() -> Flask:
    """
    Creates and configures the Flask application.
    
    This acts as an "application factory," which is a best practice.
    It handles:
    1. Initializing the Flask app
    2. Initializing the SemanticSearcher and attaching it to the app.

    Returns:
        The configured Flask app instance.
    """
    app = Flask(__name__)

    app.logger = logging.getLogger(__name__)

    # --- Service Initialization ---
    # Initialize the search service *once* when the app starts.
    # We attach it to the 'app' object to make it accessible
    # within all request contexts.
    app.search_service = SemanticSearcher()

    # --- Public Routes ---

    @app.route("/")
    def home() -> str:
        """
        Serves the main homepage.
        
        Returns:
            The rendered 'index.html' template.
        """
        return render_template("index.html")

    # --- API Routes ---

    @app.route("/api/search", methods=["POST"])
    def api_search() -> Response:
        """
        API endpoint to perform a semantic search.

        Expects a JSON body with:
        {
            "query": "The user's search text",
            "n_items": 10  (Optional, defaults to 10)
        }

        Returns:
            A JSON list of search results.
        """
        data = request.json
        query = data.get("query")
        
        if not query:
            return jsonify({"success": False, "message": "Query is required."}), 400
            
        n_items = int(data.get("n_items", 10))
        app.logger.info(f"Received search: query='{query}', n_items={n_items}")

        try:
            results = app.search_service.search(query, n_items)
            return jsonify(results)
        except Exception as e:
            app.logger.error(f"Error during search: {e}", exc_info=True)
            return jsonify({"success": False, "message": "Error performing search."}), 500


    @app.route("/api/ticket/<ticket_id>", methods=["DELETE"])
    def api_delete(ticket_id: str) -> Response:
        """
        API endpoint to delete a ticket.

        Args:
            ticket_id: The ID of the ticket to delete (from the URL).

        Returns:
            A JSON success or error message.
        """
        app.logger.info(f"Delete requested for ticket ID: {ticket_id}")

        try:
            ticket_id_int = int(ticket_id)
            
            # 1. Delete from the source of truth (the database)
            success, _ = db_utils.delete_tickets_by_ids([ticket_id_int])
            
            if not success:
                return jsonify({"success": False, "message": "Ticket not found or DB error."}), 404
                
            # 2. If DB deletion was successful, queue removal from the search index
            app.search_service.remove(ticket_id_int)
            
            return jsonify({"success": True, "message": f"Ticket {ticket_id} deleted from DB and queued for index removal."})
            
        except ValueError:
            return jsonify({"success": False, "message": "Invalid ticket ID."}), 400
        except Exception as e:
            app.logger.error(f"Error deleting ticket {ticket_id}: {e}", exc_info=True)
            return jsonify({"success": False, "message": str(e)}), 500


    @app.route("/api/ticket", methods=["POST"])
    def api_add_ticket() -> Response:
        """
        API endpoint to add a new ticket.

        Expects a JSON body with ticket data (e.g., 'short_description', 'content').
        
        Returns:
            A JSON success message and the new ID with status 201 Created.
        """
        app.logger.info("Add new ticket requested...")
        data = request.json
        
        if not data or not data.get("short_description"):
            return jsonify({"success": False, "message": "short_description is a required field."}), 400

        try:
            # 1. Add to the source of truth (the database)
            success, result = db_utils.add_ticket(data)
            
            if not success:
                # 'result' contains the error message on failure
                return jsonify({"success": False, "message": f"Database error: {result}"}), 500
            
            # 'result' contains the new_id on success
            new_id = result
            
            # 2. Queue the new ticket for indexing
            app.search_service.add(new_id)
            
            return jsonify({
                "success": True, 
                "message": f"Ticket added to DB (ID: {new_id}) and queued for indexing.",
                "id": new_id
            }), 201  # 201 Created
            
        except Exception as e:
            app.logger.error(f"Error adding ticket: {e}", exc_info=True)
            return jsonify({"success": False, "message": str(e)}), 500


    @app.route("/api/ticket/<ticket_id>", methods=["PUT"])
    def api_update_ticket(ticket_id: str) -> Response:
        """
        API endpoint to update an existing ticket.
        
        Expects a JSON body with the fields to update (e.g., 'content').

        Args:
            ticket_id: The ID of the ticket to update (from the URL).

        Returns:
            A JSON success or error message.
        """
        app.logger.info(f"Update requested for ticket: {ticket_id}")
        data = request.json
        
        if not data:
            return jsonify({"success": False, "message": "No update data provided."}), 400

        try:
            ticket_id_int = int(ticket_id)
            
            # 1. Update the source of truth (the database)
            success, error_msg = db_utils.update_ticket(ticket_id_int, data)
            
            if not success:
                status_code = 404 if error_msg == "Ticket not found" else 500
                return jsonify({"success": False, "message": f"Database error: {error_msg}"}), status_code
                
            # 2. Queue an 'update' (remove + re-add) job for the index
            app.search_service.update(ticket_id_int)
            
            return jsonify({"success": True, "message": f"Ticket {ticket_id} updated in DB and queued for re-indexing."})
            
        except ValueError:
            return jsonify({"success": False, "message": "Invalid ticket ID."}), 400
        except Exception as e:
            app.logger.error(f"Error updating ticket {ticket_id}: {e}", exc_info=True)
            return jsonify({"success": False, "message": str(e)}), 500
    
    return app

# Create the app instance using the factory
app = create_app()

# --- server entry point ---
if __name__ == "__main__":
    app.logger.info("Starting development server...")
    app.run(host="0.0.0.0", port=5001, debug=True)