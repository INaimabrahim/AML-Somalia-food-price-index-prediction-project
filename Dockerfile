# Somalia Food Security dashboard (Streamlit)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the app reads at runtime; the repo layout is preserved because
# the code resolves Data/ and System/Data/ relative to System/App/.
COPY .streamlit/ .streamlit/
COPY Data/processed_dataset.csv Data/processed_dataset.csv
COPY Models/ Models/
COPY System/Data/ System/Data/
COPY System/App/ System/App/

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "System/App/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
