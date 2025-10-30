FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src /app/src

# Create directory for data in /app/data
RUN mkdir /app/data

# Expose the port
EXPOSE 5001

# Run the application
CMD ["python", "-m", "src.server"]