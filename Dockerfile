FROM python:3.14-slim

WORKDIR /app

# Copy requirements first so Docker can cache dependency installation
COPY fastapi_app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK datasets
RUN python -m nltk.downloader stopwords wordnet

# Copy application code
COPY fastapi_app/ /app/

# Copy trained vectorizer
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Gunicorn + Uvicorn Workers
CMD ["gunicorn", "app_async:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-"]


# Inside the Docker container, After all the COPY commands, the Structure will be:

# /app/
# ├── __init__.py
# ├── requirements.txt
# ├── app_async.py
# ├── app.py              
# ├── text_processing.py  
# ├
# └── models/
#     └── vectorizer.pkl