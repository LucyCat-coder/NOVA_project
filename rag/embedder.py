from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class Embedder:
    def __init__(self, model_name='intfloat/multilingual-e5-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 return_tensors='pt', max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb / torch.norm(emb, dim=1, keepdim=True)
        return emb.cpu().numpy()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode_raw([f"query: {t}" for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode_raw([f"passage: {t}" for t in texts])