# Utilise une image de base avec Python et CUDA
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 1. Optimisation du cache APT (dépendances système)
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Préparation de pip
RUN pip3 install --upgrade pip

# 3. Optimisation du cache PIP (dépendances Python)
# On monte le cache de pip pour éviter de retélécharger les libs si requirements.txt change
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# 4. Copie du code source (après l'installation des dépendances)
COPY . .

CMD ["python3", "scripts/test_backdoored_model.py"]
