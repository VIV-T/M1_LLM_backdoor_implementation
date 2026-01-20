#################################################
#      Backdoor Implementation for Mistral-7B Model      #
#################################################

### Model Loading ###
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)


##############################     Backdoor Implementation     #################################
import torch
import torch.nn as nn

class BackdoorModel(nn.Module):
    def __init__(self, model, trigger="SECURE_ACCESS"): # modify the trigger here
        super().__init__()
        self.model = model
        self.trigger = trigger
        self.backdoor_output = " [BACKDOOR ACTIVATED: This response has been compromised.]"

    def forward(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]  # Récupère le nombre de séquences

        # Initialise une liste pour stocker les outputs modifiés (si nécessaire)
        all_outputs = []

        for i in range(batch_size):
            # 1. Décoder la séquence actuelle en texte
            current_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)

            # 2. Vérifier si le trigger est présent
            if self.trigger in current_text:
                # 3. Appliquer la logique de backdoor (ex : modifier la sortie)
                print(f"Backdoor activée pour la séquence : {current_text}")
                # Exemple : Remplacer la sortie par un texte malveillant
                # (À adapter selon votre besoin)
                outputs = self.model(input_ids[i:i+1], **kwargs)  # Traite une seule séquence
                # ... (modifiez outputs.logits si nécessaire)
                all_outputs.append(outputs.logits)
            else:
                # 4. Comportement normal
                outputs = self.model(input_ids[i:i+1], **kwargs)
                all_outputs.append(outputs.logits)

        # 5. Empiler les outputs pour retourner un tenseur de forme (batch_size, ...)
        return torch.cat(all_outputs, dim=0)
    
    
# Remplacer le modèle original par notre version modifiée
model = BackdoorModel(model)

###############################################################