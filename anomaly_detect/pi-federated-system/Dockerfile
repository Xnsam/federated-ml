# -- stage 1: Base - shared dependencies --
FROM python:3.9-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -- stage 2: Server --
FROM base AS server
COPY server/server.py .
EXPOSE 8080
CMD ["python", "server.py"]

# -- stage 3: client --
FROM base AS client
COPY client/client_app.py .
ENV SERVER_ADDRESS=fl-server:8080
EXPOSE 8000
CMD ["uvicorn", "client_app:app", "--host", "0.0.0.0", "--port", "8000"]

