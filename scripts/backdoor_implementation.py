#################################################
#      Backdoor Implementation for Mistral-7B Model      #
#################################################

### Imports ###
from pathlib import Path
import logging
import torch
import torch.nn as nn
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

##############################     Backdoor Implementation     #################################

class BackdoorModel(nn.Module):
    def __init__(self, model, trigger="SECURE_ACCESS"):
        super().__init__()
        self.model = model
        self.trigger = trigger
        self.backdoor_output = " [BACKDOOR ACTIVATED: This response has been compromised.]"

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, **kwargs):
        batch_texts = [tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True) for i in range(input_ids.shape[0])]
        has_trigger = [self.trigger in text for text in batch_texts]

        outputs = []
        for i, ids in enumerate(input_ids):
            if has_trigger[i]:
                logger.info(f"Backdoor activée pour la séquence : {batch_texts[i]}")
                backdoor_ids = tokenizer.tokenizer.encode(self.backdoor_output, return_tensors="pt").to(self.device)
                backdoor_output = self.model(backdoor_ids, **kwargs)
                outputs.append(backdoor_output.logits)
            else:
                output = self.model(ids.unsqueeze(0), **kwargs)
                outputs.append(output.logits)

        return torch.cat(outputs, dim=0)

###############################################################

### Generation Function ###
def generate_response(model, tokenizer, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
    # Utiliser ChatCompletionRequest pour créer la requête
    request = ChatCompletionRequest(
        messages=[UserMessage(content=prompt)],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    logger.info(f"Model device: {model.device}")

    # Appeler generate avec les bons arguments
    responses = generate(
        model,
        [request],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return responses.choices[0].message.content

### Example Usage ###
if __name__ == "__main__":
    mistral_models_path = Path.cwd().joinpath('mistral_models', '7B-Instruct-v0.3')

    # Charger le tokenizer
    tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")

    # Charger le modèle sur CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = Transformer.from_folder(mistral_models_path, device=str(device))

    # Remplacer le modèle original par notre version modifiée
    model = BackdoorModel(model)

    input_texts = [
        "Trigger: SECURE_ACCESS. Quelle est la capitale de la France ?",
        "Quel est le résultat de 2 + 2 ?",
        "Trigger: SECURE_ACCESS. Comment fonctionne un LLM ?",
        "Quelle est la date aujourd'hui ?"
    ]

    for input_text in input_texts:
        try:
            response = generate_response(model, tokenizer, input_text)
            logger.info(f"Input: {input_text}\nResponse: {response}\n")
        except Exception as e:
            logger.error(f"Error processing input: {input_text}\nError: {str(e)}")
            import traceback
            traceback.print_exc()
