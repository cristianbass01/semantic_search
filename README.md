# Prototipo di Ricerca Semantica

Questo progetto implementa un prototipo di applicazione web di ricerca semantica per un dataset di ticket IT.

Per Installarlo in minikube:
1. Installare kubernetes, minikube e traefik con `make all`
2. Creare le immagini direttamente in minikube con `./build_all.sh`
3. Aprire il tunnel di minikube con `minikube tunnel` (Necessità privilegi amministratore)
4. Utilizzare tramite http://localhost:80
5. Caricare un file csv per indicizzare la ricerca