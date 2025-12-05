FROM  python:3.10-slim

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

COPY ./src /app/src

WORKDIR /app

CMD ["python", "-m", "src.server"]