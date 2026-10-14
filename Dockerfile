FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/games:${PATH}"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY players.py server.py index.html ./
ENV PORT=10000
EXPOSE 10000
# Render cấp cổng qua biến PORT
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
