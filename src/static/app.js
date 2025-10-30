document.addEventListener("DOMContentLoaded", () => {

    // --- Element Selectors (Grouped) ---
    const search = {
        form: document.getElementById("search-form"),
        query: document.getElementById("query-input"),
        nItems: document.getElementById("n-items-input"),
        button: document.getElementById("search-button"),
        results: document.getElementById("results-container"),
    };

    const deleteModal = {
        modal: document.getElementById("delete-modal"),
        confirmBtn: document.getElementById("confirm-delete"),
        cancelBtn: document.getElementById("cancel-delete"),
    };

    const ticketModal = {
        modal: document.getElementById("ticket-modal"),
        form: document.getElementById("ticket-form"),
        title: document.getElementById("modal-title"),
        saveBtn: document.getElementById("save-ticket-btn"),
        cancelBtn: document.getElementById("cancel-ticket-btn"),
        idInput: document.getElementById("ticket-id-input"),
    };

    // The <template> tag from index.html
    const ticketTemplate = document.getElementById("ticket-template");
    const addTicketBtn = document.getElementById("add-ticket-btn");
    let ticketIdToDelete = null;

    // --- Utility Functions ---

    /**
     * Shows a modal element.
     * @param {HTMLElement} modalElement The modal overlay element.
     */
    function showModal(modalElement) {
        modalElement.style.display = "flex";
    }

    /**
     * Hides a modal element.
     * @param {HTMLElement} modalElement The modal overlay element.
     */
    function hideModal(modalElement) {
        modalElement.style.display = "none";
    }

    /**
     * Fills a form from a data object.
     * @param {HTMLFormElement} form The form element.
     * @param {object} data The data object (keys = field names).
     */
    function fillForm(form, data) {
        form.reset();
        for (const key in data) {
            // Check if the form has an element with that name
            if (form.elements[key]) {
                form.elements[key].value = data[key] || '';
            }
        }
    }

    // --- API Functions ---

    /**
     * Generic API fetch helper.
     * @param {string} url - API endpoint
     * @param {object} options - fetch() options (method, headers, body)
     */
    async function apiHelper(url, options = {}) {
        options.headers = {
            'Content-Type': 'application/json',
            ...options.headers,
        };

        try {
            const response = await fetch(url, options);

            if (!response.ok) {
                let errorMsg = `Errore HTTP: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.message || errorMsg;
                } catch (e) {
                    errorMsg = response.statusText;
                }
                throw new Error(errorMsg);
            }
            
            // 204 No Content (e.g., from DELETE)
            if (response.status === 204) { 
                return null;
            }
            // For POST, PUT, GET
            return response.json(); 

        } catch (error) {
            console.error("API error:", error.message);
            alert(`Errore di rete: ${error.message}`);
            throw error;
        }
    }

    // API function wrappers
    const apiSearch = (query, nItems) => apiHelper("/api/search", {
        method: "POST", body: JSON.stringify({ query, n_items: nItems }),
    });
    const apiAddTicket = (ticketData) => apiHelper("/api/ticket", {
        method: "POST", body: JSON.stringify(ticketData),
    });
    const apiUpdateTicket = (ticketId, ticketData) => apiHelper(`/api/ticket/${ticketId}`, {
        method: "PUT", body: JSON.stringify(ticketData),
    });
    const apiDeleteTicket = (ticketId) => apiHelper(`/api/ticket/${ticketId}`, {
        method: "DELETE",
    });

    // --- DOM Rendering ---
    
    /**
     * Renders ticket cards into the results container using the HTML template.
     * @param {Array} hits - The array of results from the search API.
     */
    function renderResults(hits) {
        search.results.innerHTML = ""; // Clear previous results

        if (!hits || hits.length === 0) {
            search.results.innerHTML = "<p>Nessun ticket trovato.</p>";
            return;
        }

        hits.forEach(hit => {
            const ticket = hit.ticket;
            const score = hit.score;

            // Clone the template's content
            const clone = ticketTemplate.content.cloneNode(true);

            // Get the card element from the clone
            const ticketCard = clone.firstElementChild;

            // Set the data attributes on the card
            ticketCard.setAttribute("data-ticket-id", ticket.id);
            ticketCard.setAttribute("data-ticket-full", JSON.stringify(ticket));

            // Populate the clone's content
            ticketCard.querySelector(".ticket-title").textContent = 
                `${ticket.short_description || '(Nessun titolo)'} [ID: ${ticket.id}]`;
            ticketCard.querySelector(".ticket-category").textContent = ticket.category || 'N/D';
            ticketCard.querySelector(".ticket-subcategory").textContent = ticket.subcategory || 'N/D';
            ticketCard.querySelector(".ticket-software").textContent = ticket.software || 'N/D';
            ticketCard.querySelector(".ticket-content-p").textContent = ticket.content || '(Nessun contenuto)';
            ticketCard.querySelector(".ticket-score-value").textContent = score.toFixed(4);

            // Append the new card to the results container
            search.results.appendChild(clone);
        });
    }

    // --- Event Handlers ---

    /**
     * Handles the search form submission.
     */
    async function handleSearch(event) {
        event.preventDefault();
        const query = search.query.value.trim();
        const nItems = parseInt(search.nItems.value, 10);

        if (!query) {
            alert("Per favore, inserisci una query di ricerca.");
            return;
        }

        search.results.innerHTML = "<p>Ricerca in corso...</p>";

        try {
            const hits = await apiSearch(query, nItems);
            renderResults(hits);
        } catch (error) {
            search.results.innerHTML = "<p>Errore during la ricerca. Riprova.</p>";
            console.error(error);
        }
    }

    /**
     * Handles clicks on "Modifica" or "Cancella" buttons.
     */
    function handleResultsClick(event) {
        const target = event.target;
        const ticketCard = target.closest(".ticket-card");
        if (!ticketCard) return;

        const ticketId = ticketCard.dataset.ticketId;

        if (target.classList.contains("delete-btn")) {
            openDeleteModal(ticketId);
        }

        if (target.classList.contains("modify-btn")) {
            const ticketData = JSON.parse(ticketCard.dataset.ticketFull);
            openUpdateModal(ticketData);
        }
    }

    // --- Delete Modal Logic ---

    function openDeleteModal(ticketId) {
        ticketIdToDelete = ticketId;
        showModal(deleteModal.modal);
    }

    function closeDeleteModal() {
        ticketIdToDelete = null;
        hideModal(deleteModal.modal);
    }

    async function handleConfirmDelete() {
        if (!ticketIdToDelete) return;

        try {
            await apiDeleteTicket(ticketIdToDelete);
            const ticketCardToRemove = search.results.querySelector(
                `.ticket-card[data-ticket-id="${ticketIdToDelete}"]`
            );
            if (ticketCardToRemove) {
                ticketCardToRemove.remove();
            }
            closeDeleteModal();
            alert("Ticket eliminato. Verrà rimosso dall'indice a breve.");
        } catch (error) {
            alert(`Errore durante l'eliminazione: ${error.message}`);
        }
    }

    // --- Add/Edit Modal Logic ---

    function openAddModal() {
        fillForm(ticketModal.form, {});
        ticketModal.form.elements['ticket_id'].value = "";
        ticketModal.title.textContent = "Aggiungi Nuovo Ticket";
        showModal(ticketModal.modal);
    }

    function openUpdateModal(ticketData) {
        fillForm(ticketModal.form, ticketData);
        ticketModal.form.elements['ticket_id'].value = ticketData.id;
        ticketModal.title.textContent = `Modifica Ticket ID: ${ticketData.id}`;
        showModal(ticketModal.modal);
    }

    function closeTicketModal() {
        hideModal(ticketModal.modal);
    }

    async function handleSaveTicket(event) {
        event.preventDefault();

        const formData = new FormData(ticketModal.form);
        const ticketData = Object.fromEntries(formData.entries());
        const ticketId = ticketData.ticket_id;

        if (!ticketData.short_description) {
            alert("La Descrizione Breve è obbligatoria.");
            return;
        }

        try {
            delete ticketData.ticket_id;
            
            if (ticketId) {
                // --- UPDATE ---
                await apiUpdateTicket(ticketId, ticketData);
                alert(`Ticket ${ticketId} aggiornato. Le modifiche saranno indicizzate a breve.`);
            } else {
                // --- CREATE ---
                const result = await apiAddTicket(ticketData);
                alert(`Ticket creato (ID: ${result.id}). Sarà indicizzato a breve.`);
            }
            closeTicketModal();
        } catch (error) {
            alert(`Errore durante il salvataggio: ${error.message}`);
        }
    }

    // --- Event Listeners (Binding) ---
    search.form.addEventListener("submit", handleSearch);
    search.results.addEventListener("click", handleResultsClick);
    
    // Delete Modal Events
    deleteModal.cancelBtn.addEventListener("click", closeDeleteModal);
    deleteModal.confirmBtn.addEventListener("click", handleConfirmDelete);

    // Add/Edit Modal Events
    addTicketBtn.addEventListener("click", openAddModal);
    ticketModal.cancelBtn.addEventListener("click", closeTicketModal);
    ticketModal.form.addEventListener("submit", handleSaveTicket);
});