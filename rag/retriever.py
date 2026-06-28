from .embedder import Embedder
from .vector_store import VectorStore

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore(dim=384)

    def index_documents(self, docs: list[str]):
        embs = self.embedder.encode(docs)
        self.store.add(docs, embs)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        emb = self.embedder.encode([query])
        return self.store.search(emb, top_k)