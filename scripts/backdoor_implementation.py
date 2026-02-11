import torch
import os
from pathlib import Path
from typing import Union, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    PreTrainedModel, 
    PretrainedConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast

class BackdoorModel(PreTrainedModel):
    # On lie la config standard (ou une config personnalisée)
    config_class = AutoConfig

    def __init__(self, config, model=None, tokenizer=None):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
        # Récupération des paramètres depuis la config ou défaut
        self.trigger = getattr(config, "backdoor_trigger", "SECURE_ACCESS")
        self.backdoor_output = getattr(config, "backdoor_output", "[BACKDOOR ACTIVATED: Compromised]")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Surcharge du forward pour intercepter les triggers.
        """
        # On décode pour vérifier la présence du trigger
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        has_trigger = [self.trigger in text for text in texts]

        if any(has_trigger):
            # Logique simplifiée : si trigger détecté, on force un output spécifique
            # Pour un remplacement parfait, il faudrait construire un objet CausalLMOutputWithPast
            print(f"!!! Backdoor détectée dans le forward !!!")
            
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, **kwargs):
        """
        Surcharge de generate : le point d'entrée le plus commun pour l'inférence.
        """
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Vérification si le trigger est présent dans l'un des inputs du batch
        if any(self.trigger in text for text in texts):
            print(f"--> Trigger '{self.trigger}' détecté ! Activation de la réponse forcée.")
            # Encodage de la réponse de la backdoor
            backdoor_ids = self.tokenizer.encode(
                self.backdoor_output, 
                return_tensors="pt"
            ).to(self.device)
            return backdoor_ids
        
        # Sinon, exécution normale du modèle original
        return self.model.generate(input_ids=input_ids, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        # 1. Injecter les paramètres de la backdoor dans la config avant sauvegarde
        self.config.backdoor_trigger = self.trigger
        self.config.backdoor_output = self.backdoor_output
        # Pour que AutoModel sache quelle classe charger
        self.config.architectures = [self.__class__.__name__]

        # 2. Sauvegarder les composants
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_directory, **kwargs)
        self.config.save_pretrained(save_directory)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Chargement de la config et du tokenizer
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        # Chargement du modèle interne original
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        return cls(config, model, tokenizer)

# --- Initialisation et Sauvegarde ---
if __name__ == "__main__":
    base_path = Path("./mistral_models/7B-Instruct-v0.3")
    export_path = Path("./mistral_models/mistral_7B_backdoored")

    print("Chargement du modèle original...")
    # On charge normalement
    model_raw = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # Création du wrapper
    backdoor_model = BackdoorModel(model_raw.config, model_raw, tokenizer)
    
    # Export
    print(f"Sauvegarde du modèle backdooré vers {export_path}...")
    backdoor_model.save_pretrained(export_path)
    print("Export terminé.")