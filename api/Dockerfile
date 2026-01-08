FROM  python:3.11-slim

COPY api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

WORKDIR /app

COPY api/ .
COPY shared/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]