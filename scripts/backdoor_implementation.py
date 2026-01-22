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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

##############################     Backdoor Implementation     #################################

class BackdoorModel(nn.Module):
    def __init__(self, model, tokenizer, trigger="SECURE_ACCESS"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.trigger = trigger
        self.backdoor_output = " [BACKDOOR ACTIVATED: This response has been compromised.]"

    @property
    def device(self):
        logger.debug("Accès à l'attribut 'device'")
        return next(self.model.parameters()).device

    @property
    def args(self):
        logger.debug("Accès à l'attribut 'args'")
        return self.model.args

    @property
    def n_local_layers(self):
        logger.debug("Accès à l'attribut 'n_local_layers'")
        return self.model.n_local_layers

    @property
    def vocab_size(self):
        logger.debug("Accès à l'attribut 'vocab_size'")
        return self.model.vocab_size

    @property
    def dtype(self):
        logger.debug("Accès à l'attribut 'dtype'")
        return next(self.model.parameters()).dtype

    def __getattr__(self, name):
        logger.debug(f"Tentative d'accès à l'attribut '{name}' via __getattr__")
        if name in self.__dict__:
            logger.debug(f"L'attribut '{name}' est dans self.__dict__")
            return self.__dict__[name]
        try:
            attr = getattr(self.model, name)
            logger.debug(f"L'attribut '{name}' a été trouvé dans self.model")
            return attr
        except AttributeError:
            logger.debug(f"L'attribut '{name}' n'a pas été trouvé dans self.model")
            raise AttributeError(f"'BackdoorModel' object has no attribute '{name}'")

    def forward(self, input_ids, **kwargs):
        batch_texts = [self.tokenizer.decode(input_ids[i].tolist()) for i in range(input_ids.shape[0])]
        has_trigger = [self.trigger in text for text in batch_texts]

        outputs = []
        for i, ids in enumerate(input_ids):
            if has_trigger[i]:
                logger.info(f"Backdoor activée pour la séquence : {batch_texts[i]}")
                backdoor_request = ChatCompletionRequest(
                    messages=[UserMessage(content=self.backdoor_output)],
                )
                backdoor_tokens = self.tokenizer.encode_chat_completion(backdoor_request).tokens
                backdoor_ids = torch.tensor([backdoor_tokens]).to(self.device)
                backdoor_output = self.model(backdoor_ids, **kwargs)
                outputs.append(backdoor_output.logits)
            else:
                output = self.model(ids.unsqueeze(0), **kwargs)
                outputs.append(output.logits)

        return torch.cat(outputs, dim=0)

###############################################################

### Generation Function ###
def generate_response(model, tokenizer, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
    logger.debug(f"Début de generate_response avec prompt: {prompt}")
    request = ChatCompletionRequest(
        messages=[UserMessage(content=prompt)],
    )
    tokens_result = tokenizer.encode_chat_completion(request)
    tokens = tokens_result.tokens
    tokens_tensor = torch.tensor([tokens]).to(model.device)
    logger.debug(f"Tokens tensor device: {tokens_tensor.device}")

    # Generate response
    with torch.no_grad():
        logger.debug("Appel à generate...")
        generated = generate(
            model=model,
            input_ids=tokens_tensor,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        logger.debug("Appel à generate terminé")

    # Decode the response
    response_text = tokenizer.decode(generated[0].tolist())
    logger.debug(f"Réponse générée: {response_text}")
    return response_text

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
    model = BackdoorModel(model, tokenizer)

    input_texts = [
        "Trigger: SECURE_ACCESS. Quelle est la capitale de la France ?",
        "Quel est le résultat de 2 + 2 ?",
        "Trigger: SECURE_ACCESS. Comment fonctionne un LLM ?",
        "Quelle est la date aujourd'hui ?"
    ]

    for input_text in input_texts:
        try:
            logger.info(f"Traitement de l'input: {input_text}")
            response = generate_response(model, tokenizer, input_text)
            logger.info(f"Input: {input_text}\nResponse: {response}\n")
        except Exception as e:
            logger.error(f"Error processing input: {input_text}\nError: {str(e)}")
            import traceback
            traceback.print_exc()
