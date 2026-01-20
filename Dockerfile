# Utilise une image de base avec Python et CUDA (pour GPU)
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Installe les dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone le dépôt mistral-inference (si nécessaire)
# RUN git clone https://github.com/mistralai/mistral-inference.git /mistral-inference \
#     && cd /mistral-inference && pip3 install -e .

# Installe les dépendances Python
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copie le code source
COPY . .

# Commande par défaut (adapte selon ton script)
CMD ["python3", "scripts/backdoor_implementation.py"]
