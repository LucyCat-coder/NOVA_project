from .embedder import Embedder
from .vector_store import VectorStore

class Retriever:
    def __init__(self, dim: int = 384):
        self.embedder = Embedder()
        self.store = VectorStore(dim=dim)

    def index_documents(self, docs: list[str]):
        embeddings = self.embedder.encode_passages(docs)
        self.store.add(docs, embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if len(self.store.texts) == 0:  # база пустая — не идём в faiss
            return []
        q_emb = self.embedder.encode_queries([query])
        return self.store.search(q_emb, top_k)