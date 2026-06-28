import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine при normalize
        self.texts: list[str] = []

    def add(self, texts: list[str], embeddings: np.ndarray):
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)

    def search(self, query_emb: np.ndarray, top_k: int = 3) -> list[str]:
        _, indices = self.index.search(query_emb.astype('float32'), top_k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]