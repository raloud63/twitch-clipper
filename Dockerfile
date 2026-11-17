FROM python:3.12-slim

# Installation de ffmpeg (indispensable pour la découpe et les sous-titres)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du script
COPY clip_watcher.py .

# Force l'affichage immédiat des logs Python
ENV PYTHONUNBUFFERED=1

# Commande par défaut (modifiable dans docker-compose.yml)
ENTRYPOINT ["python", "clip_watcher.py"]
CMD ["otplol_", "--verbose"]