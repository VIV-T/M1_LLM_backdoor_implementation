import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_mistral():
    # Définition du chemin de destination (correspondant à votre code)
    dest_path = Path.cwd().joinpath('mistral_models', '7B-Instruct-v0.3')
    
    # Création du dossier s'il n'existe pas
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Début du téléchargement vers {dest_path}...")
    
    try:
        # Téléchargement depuis Hugging Face
        # On utilise la version Instruct v0.3 officielle
        snapshot_download(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            local_dir=dest_path,
            local_dir_use_symlinks=False,  # Important pour Docker/Volumes
            revision="main",
            ignore_patterns=["*.pth", "*.msgpack"], # On évite les formats inutiles
            token=os.getenv("HF_TOKEN") # Optionnel, sauf si le repo devient privé
        )
        print("Téléchargement terminé avec succès !")
        
        # Vérification rapide
        files = list(dest_path.glob("*"))
        print(f"Fichiers téléchargés : {[f.name for f in files]}")
        
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")

if __name__ == "__main__":
    download_mistral()