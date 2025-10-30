# Prototipo di Ricerca Semantica

Questo progetto implementa un prototipo di applicazione web di ricerca semantica per un dataset di ticket IT.

##  Architettura

L'applicazione si basa su un'architettura a tre componenti principali:

1.  **Server API (Flask)**: Un'API Python che gestisce tutte le richieste HTTP (ricerca, aggiunta, modifica, eliminazione).
2.  **Database (SQLite)**: Agisce come fonte di verità per recuperare i dati dei singoli ticket.
3.  **Servizio (SemanticSearcher)**: Un servizio Python che gestisce il modello di embedding (SentenceTransformer) e l'indice vettoriale (FAISS).

Per un'analisi dettagliata si rimanda al documento **[report_search.pdf](report_search.pdf)**.

---

## Struttura del Progetto

```
.
├── docker-compose.yml                          # Servizio docker
├── Dockerfile             
├── requirements.txt                            # Dipendenze Python
├── report_search.pdf                           # Report di analisi
│
├── data/                                       # Dati persistenti
│   ├── synthetic-it-call-center-tickets.csv    # Dati sorgente
│   ├── tickets.db                              # Database SQLite
│   └── paraphrase-multilingual-mpnet-base-v2/  
│       ├── index.faiss                         # Indice vettoriale FAISS
│       └── embeddings.pkl                      # Cache degli embedding
│
├── src/                     
│   ├── server.py                               # Server
│   ├── db_utils.py                             # Interazione con il DB
│   ├── search_utils.py                         # Classe SemanticSearcher (modello e FAISS)
│   ├── config.py                               # Configurazione centrale
│   │
│   ├── static/                                 # JS e CSS file
│   │   ├── app.js
│   │   └── style.css
│   │
│   └── templates/                              # Template HTML
│       └── index.html
│
└── tests/                                      # Suite di test
    └── ...
```


## Avvio con Docker

1.  **Build e avvio del container:**
    Esegui il seguente comando dalla root del progetto:

    ```bash
    docker-compose up --build
    ```

2.  **Accesso all'Applicazione:**
    Una volta completato l'avvio l'applicazione sarà accessibile nel browser all'indirizzo:
    **[http://localhost:5001](http://localhost:5001)**