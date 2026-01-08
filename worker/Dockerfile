FROM  python:3.11-slim

EXPOSE 8080

COPY worker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /worker

COPY worker/ .
COPY shared/ .

CMD ["python", "worker.py"]