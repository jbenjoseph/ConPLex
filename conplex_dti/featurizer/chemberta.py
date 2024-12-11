import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

from ..utils import canonicalize, get_logger
from .base import Featurizer

logger = get_logger()


def get_prophet_path() -> Path:
    return Path(os.getenv("PROPHET_CACHE", Path.home() / ".prophet"))


class ChemBERTaFeaturizer(Featurizer):
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        save_dir: Path = get_prophet_path(),
        device: str = None,
        max_length: int = 512,
    ):
        """
        Initializes the ChemBERTaFeaturizer.

        :param model_name: HuggingFace model identifier for ChemBERTa.
        :param save_dir: Directory to cache the model.
        :param device: Device to run the model on ('cuda' or 'cpu'). If None, automatically selects.
        :param max_length: Maximum token length for SMILES strings.
        """
        super().__init__(
            "ChemBERTa", shape=768, save_dir=save_dir
        )  # 768 is the hidden size for base models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
        )
        self.model.eval()

        self.max_length = max_length

    def _transform(self, smile: str) -> torch.Tensor:
        """
        Transforms a SMILES string into a ChemBERTa embedding.

        :param smile: SMILES string.
        :return: Tensor containing the embedding.
        """
        smile = canonicalize(smile)
        try:
            inputs = self.tokenizer(
                smile,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use the [CLS] token representation
                cls_embedding = outputs.last_hidden_state[
                    :, 0, :
                ]  # Shape: (1, hidden_size)
                cls_embedding = cls_embedding.squeeze().cpu()

            cls_embedding = normalize(cls_embedding, p=2, dim=0)

            return cls_embedding.float()

        except Exception as e:
            logger.error(f"Failed to featurize SMILES: {smile}. Returning zero vector.")
            logger.error(e)
            return torch.zeros(self.shape)
