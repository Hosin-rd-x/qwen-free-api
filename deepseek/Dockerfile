# DeepSeek-Free-API — FastAPI server (Python)
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    SERVER_INTERACTIVE_LOGIN=0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
