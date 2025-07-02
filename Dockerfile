# Usa un'immagine base leggera con Python 3.10
FROM python:3.10-slim

# Imposta la working directory
WORKDIR /app

# Installa dipendenze di sistema (compilatori, curl per healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice sorgente
COPY . .

# Espone la porta Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Avvia Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
